"""Activation-clustered intervals for the WorldFloods audit.

Implements revision/rules/EVENT_CLUSTER_RULE.md. Reads committed JSON under
topics/worldfloods-cloud-aware/. Does not modify that tree, load imagery,
or run a model. Writes revision/analysis/event_cluster_results.json and
revision/supp/event_cluster_auto.tex only after the reproduction checks pass.
"""

from __future__ import annotations

import hashlib
import json
import re
import sys
import urllib.error
import urllib.request
from decimal import Decimal, ROUND_HALF_UP
from pathlib import Path

import numpy as np

ROOT = Path(__file__).resolve().parents[2]
TOPIC = ROOT / "topics" / "worldfloods-cloud-aware"
OUT_JSON = Path(__file__).resolve().parent / "event_cluster_results.json"
OUT_TEX = ROOT / "revision" / "supp" / "event_cluster_auto.tex"

N_DRAWS = 5000
PASTE_SEED = 42
MATCH_SEED = 0
ATOL_COMMITTED = 1e-6
ATOL_SINGLETON = 1e-9
PIXEL_POOL_PRINT = 2.8
API = (
    "https://rapidmapping.emergency.copernicus.eu/backend/dashboard-api/"
    "public-activations/?code={code}"
)
MAX_BYTES = 8 * 1024 * 1024
EMSR_RE = re.compile(r"^EMSR\d+")
EMSR_TOKEN = re.compile(r"EMSR\d+")


def load(name):
    path = TOPIC / name
    return json.loads(path.read_text()), sha256(path)


def sha256(path: Path) -> str:
    h = hashlib.sha256()
    h.update(path.read_bytes())
    return h.hexdigest()


def cluster_id(scene: str) -> str:
    match = EMSR_RE.match(scene)
    if match:
        return match.group(0)
    return scene


def finite_pairs(rows, value):
    out = []
    for row in rows:
        gap = value(row)
        if gap is None or not np.isfinite(gap):
            continue
        out.append((cluster_id(row["scene"]), float(gap), row["scene"]))
    return out


def scene_bootstrap(gaps, seed):
    gaps = np.asarray(gaps, dtype=float)
    rng = np.random.default_rng(seed)
    draws = [float(rng.choice(gaps, len(gaps)).mean()) for _ in range(N_DRAWS)]
    lo, hi = np.percentile(draws, [2.5, 97.5])
    return [float(lo), float(hi)]


def _buckets(pairs):
    order = []
    buckets = {}
    for cid, gap, _scene in pairs:
        if cid not in buckets:
            order.append(cid)
            buckets[cid] = []
        buckets[cid].append(gap)
    return order, buckets


def cluster_bootstrap(pairs, seed, weighted):
    order, buckets = _buckets(pairs)
    n_clusters = len(order)
    rng = np.random.default_rng(seed)
    draws = []
    for _ in range(N_DRAWS):
        picked = rng.choice(n_clusters, size=n_clusters, replace=True)
        if weighted:
            draws.append(float(np.mean([np.mean(buckets[order[i]]) for i in picked])))
        else:
            vals = []
            for i in picked:
                vals.extend(buckets[order[i]])
            draws.append(float(np.mean(vals)))
    lo, hi = np.percentile(draws, [2.5, 97.5])
    return [float(lo), float(hi)]


def summarize(pairs, seed, committed_mean, committed_ci):
    gaps = [gap for _cid, gap, _scene in pairs]
    order, buckets = _buckets(pairs)
    scene_mean = float(np.mean(gaps))
    event_mean = float(np.mean([np.mean(buckets[c]) for c in order]))
    scene_ci = scene_bootstrap(gaps, seed)
    cluster_ci = cluster_bootstrap(pairs, seed, weighted=False)
    event_ci = cluster_bootstrap(pairs, seed, weighted=True)
    if abs(scene_mean - committed_mean) > ATOL_COMMITTED:
        raise SystemExit(
            f"scene mean {scene_mean} != committed {committed_mean}"
        )
    if not np.allclose(scene_ci, committed_ci, rtol=0, atol=ATOL_COMMITTED):
        raise SystemExit(f"scene interval {scene_ci} != committed {committed_ci}")
    sizes = [len(buckets[c]) for c in order]
    if all(n == 1 for n in sizes):
        if not np.allclose(cluster_ci, scene_ci, rtol=0, atol=ATOL_SINGLETON):
            raise SystemExit(
                f"singleton cluster interval {cluster_ci} != scene {scene_ci}"
            )
    return dict(
        scenes=len(gaps),
        clusters=len(order),
        cluster_sizes=sizes,
        max_cluster_size=int(max(sizes)),
        scene_mean=scene_mean,
        event_weighted_mean=event_mean,
        scene_bootstrap_ci95=scene_ci,
        cluster_bootstrap_ci95=cluster_ci,
        event_weighted_ci95=event_ci,
        committed_ci95=list(committed_ci),
    )


def paste_stats(blobs):
    p2 = blobs["p2_results.json"]["rows"]
    sh = blobs["p2_shuffled_results.json"]["rows"]
    soft = blobs["p2_soft_edge_results.json"]["rows"]
    var = blobs["p2_varying_results.json"]["rows"]
    own = lambda r: bool(r["own_mask"]) and bool(r["own_spectrum"])
    specs = [
        ("hard_all", "Hard constant paste", p2, lambda r: r["gap_pts"], None),
        ("hard_val", "Hard constant paste, validation", p2, lambda r: r["gap_pts"], "val"),
        ("hard_test", "Hard constant paste, test", p2, lambda r: r["gap_pts"], "test"),
        ("own", "Own mask and spectrum", p2, lambda r: r["gap_pts"] if own(r) else None, None),
        ("borrowed", "Borrowed or frozen", p2, lambda r: None if own(r) else r["gap_pts"], None),
        ("shuffled", "Shuffled 16 px blocks", sh, lambda r: r["gap_sh"], None),
        ("paired_diff", "Real minus shuffled", sh, lambda r: r["gap_real"] - r["gap_sh"], None),
        ("bin_0_8", "Hard paste, within 8 px", sh, lambda r: r["gap_real_bin0"], None),
        ("bin_8_32", "Hard paste, 8 to 32 px", sh, lambda r: r["gap_real_bin1"], None),
        ("bin_gt_32", "Hard paste, beyond 32 px", sh, lambda r: r["gap_real_bin2"], None),
        ("soft", "Soft edge", soft, lambda r: r["gap_soft"], None),
        ("varying", "Per-pixel spectrum, soft edge", var, lambda r: r["gap_soft_var"], None),
    ]
    committed = committed_paste(blobs)
    out = {}
    for key, label, rows, value, split in specs:
        use = rows if split is None else [r for r in rows if r["split"] == split]
        pairs = finite_pairs(use, value)
        mean, ci = committed[key]
        out[key] = dict(label=label, seed=PASTE_SEED, **summarize(pairs, PASTE_SEED, mean, ci))
    return out


def committed_paste(blobs):
    a = blobs["p2_analyze.out.json"]
    f = blobs["p2_followup_analyze.out.json"]
    s = blobs["p2_soft_edge_analyze.out.json"]
    v = blobs["p2_varying_analyze.out.json"]

    def pull(block, mean_key="mean"):
        return block[mean_key], block["ci95"]

    return {
        "hard_all": pull(a["headline_val_test"], "mean_gap_pts"),
        "hard_val": pull(a["val"], "mean_gap_pts"),
        "hard_test": pull(a["test"], "mean_gap_pts"),
        "own": pull(f["own_cloud_only"]),
        "borrowed": pull(f["complement"]),
        "shuffled": pull(f["shuffled"]),
        "paired_diff": pull(f["paired_real_minus_shuffled"]),
        "bin_0_8": pull(f["edge_real_0-8"]),
        "bin_8_32": pull(f["edge_real_8-32"]),
        "bin_gt_32": pull(f["edge_real_>32"]),
        "soft": pull(s["soft"]),
        "varying": pull(v["soft_var"]),
    }


def match_stats(blobs):
    kill = blobs["kill_test_results.json"]["test_val"]
    water = blobs["method1_results.json"]["terrain"]
    ndvi = blobs["ndvi_comatch_results.json"]["ndvi_comatch"]
    specs = [
        ("unmatched", "Unmatched clear versus cloudy", kill["per_scene"], "gap", kill["summary"]["mean_gap_pts"], kill["summary"]["ci95"]),
        ("water", "Water-fraction match", water["matched"], "gap", water["matched_mean_gap"], water["matched_ci95"]),
        ("ndvi", "NDVI co-match", ndvi["per_scene"], "gap", ndvi["mean_gap"], ndvi["ci95"]),
    ]
    out = {}
    for key, label, rows, field, mean, ci in specs:
        pairs = finite_pairs(rows, lambda r, field=field: r[field])
        out[key] = dict(label=label, seed=MATCH_SEED, **summarize(pairs, MATCH_SEED, mean, ci))
    return out


def pool(tiles, key="base"):
    if not tiles:
        return np.zeros(3)
    return np.array([t["cm"][key] for t in tiles], dtype=float).sum(0)


def iou(counts):
    return counts[0] / max(counts.sum(), 1.0)


def replay_accounting(tiles, stored_water, stored_ndvi):
    by = {}
    for tile in tiles:
        by.setdefault(tile["scene"], []).append(tile)
    paired = []
    for scene, ts in by.items():
        clear = pool([t for t in ts if t["cloud_frac"] < 0.01])
        cloudy = pool([t for t in ts if t["cloud_frac"] >= 0.1])
        if clear[0] + clear[2] >= 200 and cloudy[0] + cloudy[2] >= 200:
            paired.append(scene)
    water_rows = []
    ndvi_rows = []
    for scene in paired:
        ts = by[scene]
        clear = [t for t in ts if t["cloud_frac"] < 0.01]
        cloudy = [t for t in ts if t["cloud_frac"] >= 0.1]
        if any(t["water_frac"] is None for t in clear + cloudy):
            raise SystemExit(f"null water fraction in {scene}")
        used = set()
        matched_clear = []
        matched_cloudy = []
        for tile in sorted(cloudy, key=lambda t: t["water_frac"]):
            cand = [
                (abs(u["water_frac"] - tile["water_frac"]), i)
                for i, u in enumerate(clear)
                if i not in used
            ]
            if cand and min(cand)[0] <= 0.05:
                i = min(cand)[1]
                used.add(i)
                matched_clear.append(clear[i])
                matched_cloudy.append(tile)
        pa, pb = pool(matched_clear), pool(matched_cloudy)
        in_mean = bool(pa[0] + pa[2] >= 200 and pb[0] + pb[2] >= 200 and matched_cloudy)
        gap = 100 * (iou(pa) - iou(pb)) if in_mean else None
        water_rows.append(
            dict(
                scene=scene,
                cloudy_available=len(cloudy),
                cloudy_matched=len(matched_cloudy),
                cloudy_discarded=len(cloudy) - len(matched_cloudy),
                clear_available=len(clear),
                clear_used=len(used),
                in_mean=in_mean,
                gap=gap,
            )
        )
        clear_n = [t for t in clear if t.get("ndvi_clear") is not None]
        cloudy_n = [t for t in cloudy if t.get("ndvi_clear") is not None]
        skipped = len(cloudy) - len(cloudy_n)
        used_n = set()
        mc = []
        md = []
        for tile in sorted(cloudy_n, key=lambda t: t["water_frac"]):
            cand = []
            for i, u in enumerate(clear_n):
                if i in used_n:
                    continue
                dw = abs(u["water_frac"] - tile["water_frac"])
                dn = abs(u["ndvi_clear"] - tile["ndvi_clear"])
                if dw <= 0.05 and dn <= 0.05:
                    cand.append((dw, i))
            if cand:
                i = min(cand)[1]
                used_n.add(i)
                mc.append(clear_n[i])
                md.append(tile)
        pa, pb = pool(mc), pool(md)
        in_mean_n = bool(pa[0] + pa[2] >= 200 and pb[0] + pb[2] >= 200 and md)
        gap_n = 100 * (iou(pa) - iou(pb)) if in_mean_n else None
        ndvi_rows.append(
            dict(
                scene=scene,
                cloudy_available=len(cloudy),
                cloudy_skipped_missing_ndvi=skipped,
                cloudy_matched=len(md),
                cloudy_discarded=len(cloudy_n) - len(md),
                clear_used=len(used_n),
                in_mean=in_mean_n,
                gap=gap_n,
            )
        )
    check_replay(water_rows, stored_water, "water")
    check_replay(ndvi_rows, stored_ndvi, "ndvi")
    return water_rows, ndvi_rows


def check_replay(rows, stored, name):
    got = {r["scene"]: r for r in rows if r["in_mean"]}
    want = {r["scene"]: r for r in stored}
    if set(got) != set(want):
        raise SystemExit(f"{name} replay scenes {sorted(got)} != stored {sorted(want)}")
    for scene, src in want.items():
        if got[scene]["cloudy_matched"] != src["pairs"]:
            raise SystemExit(f"{name} {scene} pairs {got[scene]['cloudy_matched']} != {src['pairs']}")
        if abs(got[scene]["gap"] - src["gap"]) > ATOL_COMMITTED:
            raise SystemExit(f"{name} {scene} gap {got[scene]['gap']} != {src['gap']}")


def round_half_up(x, nd):
    quantum = Decimal("1").scaleb(-nd)
    return Decimal(str(x)).quantize(quantum, rounding=ROUND_HALF_UP)


def display_bound(x):
    one = float(round_half_up(x, 1))
    crosses = (x > 0 >= one) or (x < 0 <= one) or (x > 5 >= one) or (x < 5 <= one)
    if crosses:
        text = f"{float(round_half_up(x, 3)):.3f}"
    else:
        text = f"{one:.1f}"
    if text.startswith("-"):
        return "$-$" + text[1:]
    return text


def display_ci(ci):
    return f"[{display_bound(ci[0])}, {display_bound(ci[1])}]"


def esc(text):
    return (
        text.replace("\\", r"\textbackslash{}")
        .replace("&", r"\&")
        .replace("%", r"\%")
        .replace("_", r"\_")
        .replace("#", r"\#")
    )


def tex_rows(stats):
    lines = []
    for key in stats:
        row = stats[key]
        lines.append(
            " & ".join(
                [
                    row["label"],
                    str(row["scenes"]),
                    str(row["clusters"]),
                    display_bound(row["scene_mean"]),
                    display_bound(row["event_weighted_mean"]),
                    display_ci(row["cluster_bootstrap_ci95"]),
                ]
            )
            + r" \\"
        )
    return "\n".join(lines)


def tex_accounting(rows, ndvi=False):
    lines = []
    for row in rows:
        scene = r"\texttt{" + esc(row["scene"]) + "}"
        cells = [
            scene,
            str(row["cloudy_available"]),
            str(row["cloudy_matched"]),
            str(row["cloudy_discarded"]),
            str(row["clear_used"]),
            "yes" if row["in_mean"] else "no",
        ]
        if ndvi:
            cells.insert(2, str(row["cloudy_skipped_missing_ndvi"]))
        lines.append(" & ".join(cells) + r" \\")
    return "\n".join(lines)


def reading(paste):
    hard = paste["hard_all"]
    margin_threshold = hard["scene_mean"] - PIXEL_POOL_PRINT
    margin = hard["scene_mean"] - hard["event_weighted_mean"]
    lower_le_0 = hard["cluster_bootstrap_ci95"][0] <= 0
    move_caveat = margin >= margin_threshold
    return dict(
        hard_cluster_lower=hard["cluster_bootstrap_ci95"][0],
        hard_cluster_lower_at_most_0=lower_le_0,
        scene_mean=hard["scene_mean"],
        event_weighted_mean=hard["event_weighted_mean"],
        margin_scene_minus_event=margin,
        margin_threshold=margin_threshold,
        pixel_pool_print=PIXEL_POOL_PRINT,
        move_aggregator_caveat=move_caveat,
    )


def prose(paste, matches, read):
    hard = paste["hard_all"]
    interval = display_ci(hard["cluster_bootstrap_ci95"])
    if read["hard_cluster_lower_at_most_0"]:
        claim = (
            "The 32-scene hard-paste mean stays positive under scene resampling "
            "and is not distinguishable from 0 under event resampling. "
            f"The activation-clustered interval is {interval}."
        )
    else:
        claim = (
            "Activation-clustered intervals leave the 32-scene hard-paste conclusion unchanged: "
            f"the lower bound stays above 0. The scene mean is {display_bound(hard['scene_mean'])} "
            f"and the activation-clustered interval is {interval} "
            f"({hard['clusters']} activation clusters)."
        )
    if read["move_aggregator_caveat"]:
        caveat = (
            f"The event-weighted mean is {display_bound(hard['event_weighted_mean'])}, "
            "at least 2.8 points below the scene mean, so the aggregator caveat belongs in the claim."
        )
    else:
        caveat = (
            f"The event-weighted mean is {display_bound(hard['event_weighted_mean'])}. "
            "The difference from the scene mean is below the pre-committed margin of 2.8 points, "
            "so the aggregator caveat stays with the paste results."
        )
    parts = [claim, caveat]
    for key in ("unmatched", "water", "ndvi"):
        row = matches[key]
        if row["max_cluster_size"] != 1:
            parts.append(
                f"{row['label']} is not one scene per activation "
                f"(largest cluster {row['max_cluster_size']})."
            )
        else:
            parts.append(
                f"{row['label']} has {row['scenes']} scenes in {row['clusters']} activations, "
                "so the clustered interval equals the committed scene bootstrap."
            )
    return " ".join(parts)


def write_tex(paste, matches, water_rows, ndvi_rows, read, sensitivity):
    sens_note = sensitivity["note"]
    body = r"""\section{Activation-clustered intervals}
\label{sec:s_cluster}
This section is additional analysis of the committed per-scene JSON. It is not a new experiment. The paste remains a constructed fill on clear tiles, scored on unpainted pixels. It is not a table of observed cloud. The rule is \texttt{revision/rules/EVENT\_CLUSTER\_RULE.md}, committed before these intervals were computed. A cluster is the EMSR activation code, and each non-EMSR scene is its own cluster. The draw resamples clusters, 5000 times, with the same seeds as the scene bootstrap (42 for the paste family, 0 for the matches). The scene mean is unchanged by construction. The scene-bootstrap intervals were recomputed only as a check and matched the committed files; they are not reprinted.

""" + prose(paste, matches, read) + "\n\n"
    body += r"""\begin{table}[h]
\centering
\scriptsize
\setlength{\tabcolsep}{4pt}
\caption{Activation-clustered intervals for the locked scene means. The interval resamples activations. The event-weighted mean averages within an activation, then across activations.}
\label{tab:s_cluster}
\begin{tabular}{lrrrrl}
\toprule
Statistic & Scenes & Clusters & Scene mean & Event-weighted & Cluster interval \\
\midrule
"""
    body += tex_rows(paste) + "\n" + tex_rows(matches) + "\n"
    body += r"""\bottomrule
\end{tabular}
\end{table}

"""
    body += sens_note + "\n\n"
    body += r"""\begin{table}[h]
\centering
\scriptsize
\setlength{\tabcolsep}{3pt}
\caption{Water-fraction match accounting, replayed from \texttt{method1\_tiles.json} with the committed matcher. A scene with no pair is still a row. ``In the mean'' is the 200-pixel gate.}
\label{tab:s_match_water}
\begin{tabular}{lrrrrl}
\toprule
Scene & Cloudy & Matched & Discarded & Clear used & In the mean \\
\midrule
"""
    body += tex_accounting(water_rows) + "\n"
    body += r"""\bottomrule
\end{tabular}
\end{table}

\begin{table}[h]
\centering
\scriptsize
\setlength{\tabcolsep}{3pt}
\caption{NDVI co-match accounting on the same scenes. Skipped tiles have no NDVI. Discarded tiles have an NDVI and no partner inside both tolerances.}
\label{tab:s_match_ndvi}
\begin{tabular}{lrrrrrl}
\toprule
Scene & Cloudy & Skipped & Matched & Discarded & Clear used & In the mean \\
\midrule
"""
    body += tex_accounting(ndvi_rows, ndvi=True) + "\n"
    body += r"""\bottomrule
\end{tabular}
\end{table}
"""
    OUT_TEX.write_text(body)


def fetch_activation(code):
    url = API.format(code=code)
    req = urllib.request.Request(url, headers={"User-Agent": "worldfloods-audit-reanalysis"})
    try:
        with urllib.request.urlopen(req, timeout=30) as resp:
            raw = resp.read(MAX_BYTES + 1)
    except (urllib.error.URLError, TimeoutError, OSError) as exc:
        return None, f"{code}: {exc}"
    if len(raw) > MAX_BYTES:
        return None, f"{code}: response exceeds 8 MB"
    try:
        payload = json.loads(raw.decode("utf-8"))
    except json.JSONDecodeError as exc:
        return None, f"{code}: {exc}"
    return payload, None


def activation_record(payload, code):
    results = payload.get("results", payload if isinstance(payload, list) else None)
    if isinstance(payload, dict) and "code" in payload and results is None:
        results = [payload]
    if not isinstance(results, list):
        return None
    for item in results:
        if isinstance(item, dict) and item.get("code") == code:
            return item
    if len(results) == 1 and isinstance(results[0], dict):
        return results[0]
    return None


class UnionFind:
    def __init__(self, items):
        self.parent = {item: item for item in items}

    def find(self, item):
        while self.parent[item] != item:
            self.parent[item] = self.parent[self.parent[item]]
            item = self.parent[item]
        return item

    def union(self, a, b):
        if a not in self.parent or b not in self.parent:
            return
        ra, rb = self.find(a), self.find(b)
        if ra != rb:
            self.parent[rb] = ra


def sensitivity(codes):
    records = {}
    for code in sorted(codes):
        payload, err = fetch_activation(code)
        if err:
            return dict(ok=False, note=r"The sensitivity grouping was not computed. " + esc(err) + ".", error=err)
        if "gdacsId" not in json.dumps(payload) or "relatedevents" not in json.dumps(payload):
            reason = f"{code}: gdacsId or relatedevents absent"
            return dict(ok=False, note="The sensitivity grouping was not computed. The catalogue response has no shared-event field this rule accepts.", error=reason)
        rec = activation_record(payload, code)
        if rec is None:
            reason = f"{code}: activation record not found"
            return dict(ok=False, note=r"The sensitivity grouping was not computed. " + esc(reason) + ".", error=reason)
        related_text = json.dumps(rec.get("relatedevents"))
        records[code] = dict(
            gdacsId=rec.get("gdacsId") or None,
            related=sorted(set(EMSR_TOKEN.findall(related_text))),
        )
    uf = UnionFind(codes)
    by_gdacs = {}
    for code, rec in records.items():
        gid = rec["gdacsId"]
        if gid:
            by_gdacs.setdefault(gid, []).append(code)
        for other in rec["related"]:
            if other in uf.parent and other != code:
                uf.union(code, other)
    for group in by_gdacs.values():
        for other in group[1:]:
            uf.union(group[0], other)
    parent = {code: uf.find(code) for code in codes}
    merged = sorted({code for code in codes if parent[code] != code or list(parent.values()).count(parent[code]) > 1})
    groups = {}
    for code, root in parent.items():
        groups.setdefault(root, []).append(code)
    merged_groups = [sorted(v) for v in groups.values() if len(v) > 1]
    if not merged_groups:
        note = "The sensitivity grouping found no two activations that share a \\texttt{gdacsId} or list each other under \\texttt{relatedevents}. No second interval is reported."
    else:
        shown = "; ".join(", ".join(g) for g in merged_groups)
        note = "The sensitivity grouping merges " + esc(shown) + ". Intervals under that grouping are in the result JSON and are not the primary intervals."
    return dict(ok=True, note=note, parent=parent, records=records, merged_groups=merged_groups, merged_codes=merged)


def remap_pairs(pairs, parent):
    out = []
    for cid, gap, scene in pairs:
        if cid in parent:
            cid = parent[cid]
        out.append((cid, gap, scene))
    return out


def sensitivity_intervals(paste_pairs, match_pairs, parent):
    out = {}
    for key, pairs, seed in paste_pairs:
        mapped = remap_pairs(pairs, parent)
        order, buckets = _buckets(mapped)
        out[key] = dict(
            clusters=len(order),
            max_cluster_size=int(max(len(buckets[c]) for c in order)),
            event_weighted_mean=float(np.mean([np.mean(buckets[c]) for c in order])),
            cluster_bootstrap_ci95=cluster_bootstrap(mapped, seed, weighted=False),
        )
    for key, pairs, seed in match_pairs:
        mapped = remap_pairs(pairs, parent)
        order, buckets = _buckets(mapped)
        out[key] = dict(
            clusters=len(order),
            max_cluster_size=int(max(len(buckets[c]) for c in order)),
            cluster_bootstrap_ci95=cluster_bootstrap(mapped, seed, weighted=False),
        )
    return out


def collect_pairs(blobs):
    p2 = blobs["p2_results.json"]["rows"]
    sh = blobs["p2_shuffled_results.json"]["rows"]
    soft = blobs["p2_soft_edge_results.json"]["rows"]
    var = blobs["p2_varying_results.json"]["rows"]
    own = lambda r: bool(r["own_mask"]) and bool(r["own_spectrum"])
    paste = [
        ("hard_all", finite_pairs(p2, lambda r: r["gap_pts"]), PASTE_SEED),
        ("hard_val", finite_pairs([r for r in p2 if r["split"] == "val"], lambda r: r["gap_pts"]), PASTE_SEED),
        ("hard_test", finite_pairs([r for r in p2 if r["split"] == "test"], lambda r: r["gap_pts"]), PASTE_SEED),
        ("own", finite_pairs(p2, lambda r: r["gap_pts"] if own(r) else None), PASTE_SEED),
        ("borrowed", finite_pairs(p2, lambda r: None if own(r) else r["gap_pts"]), PASTE_SEED),
        ("shuffled", finite_pairs(sh, lambda r: r["gap_sh"]), PASTE_SEED),
        ("paired_diff", finite_pairs(sh, lambda r: r["gap_real"] - r["gap_sh"]), PASTE_SEED),
        ("bin_0_8", finite_pairs(sh, lambda r: r["gap_real_bin0"]), PASTE_SEED),
        ("bin_8_32", finite_pairs(sh, lambda r: r["gap_real_bin1"]), PASTE_SEED),
        ("bin_gt_32", finite_pairs(sh, lambda r: r["gap_real_bin2"]), PASTE_SEED),
        ("soft", finite_pairs(soft, lambda r: r["gap_soft"]), PASTE_SEED),
        ("varying", finite_pairs(var, lambda r: r["gap_soft_var"]), PASTE_SEED),
    ]
    kill = blobs["kill_test_results.json"]["test_val"]["per_scene"]
    water = blobs["method1_results.json"]["terrain"]["matched"]
    ndvi = blobs["ndvi_comatch_results.json"]["ndvi_comatch"]["per_scene"]
    matches = [
        ("unmatched", finite_pairs(kill, lambda r: r["gap"]), MATCH_SEED),
        ("water", finite_pairs(water, lambda r: r["gap"]), MATCH_SEED),
        ("ndvi", finite_pairs(ndvi, lambda r: r["gap"]), MATCH_SEED),
    ]
    return paste, matches


def codes_in(pairs_groups):
    found = set()
    for _key, pairs, _seed in pairs_groups:
        for cid, _gap, _scene in pairs:
            if cid.startswith("EMSR"):
                found.add(cid)
    return found


def main():
    names = [
        "p2_results.json",
        "p2_shuffled_results.json",
        "p2_soft_edge_results.json",
        "p2_varying_results.json",
        "p2_analyze.out.json",
        "p2_followup_analyze.out.json",
        "p2_soft_edge_analyze.out.json",
        "p2_varying_analyze.out.json",
        "kill_test_results.json",
        "method1_results.json",
        "ndvi_comatch_results.json",
        "method1_tiles.json",
    ]
    blobs = {}
    hashes = {}
    for name in names:
        blobs[name], hashes[name] = load(name)
    paste = paste_stats(blobs)
    matches = match_stats(blobs)
    water_rows, ndvi_rows = replay_accounting(
        blobs["method1_tiles.json"],
        blobs["method1_results.json"]["terrain"]["matched"],
        blobs["ndvi_comatch_results.json"]["ndvi_comatch"]["per_scene"],
    )
    read = reading(paste)
    paste_pairs, match_pairs = collect_pairs(blobs)
    try:
        sens = sensitivity(codes_in(paste_pairs + match_pairs))
    except Exception as exc:
        sens = dict(
            ok=False,
            note="The sensitivity grouping was not computed. " + esc(str(exc)) + ".",
            error=str(exc),
        )
    if sens["ok"] and sens["merged_groups"]:
        sens["intervals"] = sensitivity_intervals(paste_pairs, match_pairs, sens["parent"])
    result = dict(
        rule="revision/rules/EVENT_CLUSTER_RULE.md",
        n_draws=N_DRAWS,
        numpy=np.__version__,
        input_sha256=hashes,
        paste=paste,
        matches=matches,
        reading=read,
        water_accounting=water_rows,
        ndvi_accounting=ndvi_rows,
        sensitivity={k: v for k, v in sens.items() if k != "note"},
        sensitivity_note=sens["note"],
    )
    OUT_JSON.write_text(json.dumps(result, indent=1) + "\n")
    write_tex(paste, matches, water_rows, ndvi_rows, read, sens)
    print(json.dumps({"reading": read, "sensitivity_ok": sens["ok"], "out": str(OUT_JSON)}, indent=1))


if __name__ == "__main__":
    main()
