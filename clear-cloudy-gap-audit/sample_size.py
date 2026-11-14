"""Interval width at hypothetical sample sizes, from committed scene gaps.

Reads topics/worldfloods-cloud-aware/. Does not modify that tree, load
imagery, or run a model. n=20 and n=50 are i.i.d. resamples of the gaps
already stored. They are not new experiments and not new point estimates.
"""

from __future__ import annotations

import json
from pathlib import Path

import numpy as np

ROOT = Path(__file__).resolve().parents[2]
TOPIC = ROOT / "topics" / "worldfloods-cloud-aware"
OUT_JSON = Path(__file__).resolve().parent / "sample_size_results.json"
OUT_TEX = ROOT / "revision" / "supp" / "sample_size_auto.tex"

N_DRAWS = 5000
HYPOTHETICAL = (20, 50)
ATOL = 1e-6


def load(name: str):
    return json.loads((TOPIC / name).read_text())


def percentile_interval(values: np.ndarray, n: int, seed: int) -> tuple[float, float]:
    rng = np.random.default_rng(seed)
    draws = np.empty(N_DRAWS, dtype=float)
    for i in range(N_DRAWS):
        draws[i] = float(rng.choice(values, size=n, replace=True).mean())
    lo, hi = np.percentile(draws, [2.5, 97.5])
    return float(lo), float(hi)


def width(lo: float, hi: float) -> float:
    return float(hi - lo)


def one_decimal(x: float) -> str:
    # Display only. Full precision stays in the JSON.
    return f"{x:.1f}"


def tex_escape(text: str) -> str:
    return text.replace("_", r"\_")


def family_row(label: str, values: list[float], seed: int, stored_ci: list[float]) -> dict:
    arr = np.asarray(values, dtype=float)
    n_obs = int(arr.size)
    lo_obs, hi_obs = percentile_interval(arr, n_obs, seed)
    if abs(lo_obs - stored_ci[0]) > ATOL or abs(hi_obs - stored_ci[1]) > ATOL:
        raise SystemExit(
            f"{label}: reproduced interval [{lo_obs}, {hi_obs}] "
            f"does not match stored [{stored_ci[0]}, {stored_ci[1]}]"
        )
    hypothetical = {}
    for n in HYPOTHETICAL:
        lo, hi = percentile_interval(arr, n, seed)
        hypothetical[str(n)] = {
            "n": n,
            "lo": lo,
            "hi": hi,
            "width": width(lo, hi),
            "label": "extrapolation",
        }
    return {
        "label": label,
        "n_observed": n_obs,
        "seed": seed,
        "draws": N_DRAWS,
        "stored_ci95": [float(stored_ci[0]), float(stored_ci[1])],
        "stored_width": width(float(stored_ci[0]), float(stored_ci[1])),
        "reproduced_ci95": [lo_obs, hi_obs],
        "hypothetical": hypothetical,
    }


def inventory() -> dict:
    tiles = load("method1_tiles.json")
    cloud = load("cloud_iou_tiles.json")
    p2 = load("p2_results.json")["rows"]
    tile_ids = {t["scene"] for t in tiles}
    cloud_ids = {t["scene"] for t in cloud}
    p2_ids = {r["scene"] for r in p2}
    missing = sorted(tile_ids - p2_ids)
    by_split: dict[str, set[str]] = {}
    tile_counts: dict[str, int] = {}
    for t in tiles:
        by_split.setdefault(t["split"], set()).add(t["scene"])
        tile_counts[t["scene"]] = tile_counts.get(t["scene"], 0) + 1
    return {
        "method1_tiles": len(tiles),
        "method1_scenes": len(tile_ids),
        "method1_scenes_by_split": {k: len(v) for k, v in sorted(by_split.items())},
        "cloud_iou_tiles": len(cloud),
        "cloud_iou_scenes": len(cloud_ids),
        "same_scene_ids": tile_ids == cloud_ids,
        "paste_scenes": len(p2_ids),
        "paste_by_split": {
            "val": sum(1 for r in p2 if r["split"] == "val"),
            "test": sum(1 for r in p2 if r["split"] == "test"),
        },
        "scenes_with_tiles_and_no_paste_gap": [
            {"scene": s, "tiles": tile_counts[s]} for s in missing
        ],
    }


def write_tex(info: dict, rows: list[dict]) -> None:
    missing = info["scenes_with_tiles_and_no_paste_gap"]
    missing_tex = ", ".join(
        f"\\texttt{{{tex_escape(m['scene'])}}} ({m['tiles']} tiles)" for m in missing
    )
    lines = [
        r"\section{Power and sample size}",
        r"\label{sec:s_power}",
        (
            "This section counts the scenes already stored and then resamples those "
            "scene gaps. It does not add a scene, and it does not report a new point estimate. "
            f"\\texttt{{method1\\_tiles.json}} has {info['method1_tiles']} tiles and "
            f"{info['method1_scenes']} scenes "
            f"({info['method1_scenes_by_split']['val']} validation, "
            f"{info['method1_scenes_by_split']['test']} test). "
            f"\\texttt{{cloud\\_iou\\_tiles.json}} has the same {info['cloud_iou_tiles']} tiles "
            f"and the same {info['cloud_iou_scenes']} scene identifiers. "
            f"A stored clear-versus-cloudy gap exists for 8 of those scenes "
            f"(\\texttt{{kill\\_test\\_results.json}}, \\texttt{{test\\_val.per\\_scene}}). "
            f"A stored water-fraction gap exists for 7 "
            f"(\\texttt{{method1\\_results.json}}, \\texttt{{terrain.matched}}). "
            f"A stored NDVI co-match gap exists for 5 "
            f"(\\texttt{{ndvi\\_comatch\\_results.json}}, \\texttt{{ndvi\\_comatch.per\\_scene}}). "
            f"A stored hard-paste gap exists for {info['paste_scenes']} "
            f"({info['paste_by_split']['val']} validation, {info['paste_by_split']['test']} test) "
            f"in \\texttt{{p2\\_results.json}}. "
            f"The other tile scenes have no row there. They are {missing_tex}. "
            "No gap for either scene is stored, so neither scene is added here."
        ),
        "",
        (
            "Table~\\ref{tab:s_power} uses only those stored per-scene gaps. "
            "For a target $n$, the draw takes $n$ gaps with replacement from that scene list, "
            "averages them, and repeats the draw 5000 times. The width is the distance between "
            "the 2.5 and 97.5 percentiles of those averages. "
            "The unmatched, water-fraction, and NDVI draws use seed 0. The hard-paste draw uses seed 42. "
            "Each cell has its own generator. At the observed $n$ the interval matches the committed "
            "\\texttt{ci95} to $10^{-6}$, and that check is the only use of the observed column. "
            "$n{=}20$ and $n{=}50$ are extrapolations. They assume future scenes are i.i.d.\\ draws "
            "from the empirical distribution of the scenes already scored. "
            "They are not a new experiment, and the table prints no mean for them."
        ),
        "",
        r"\begin{table}[h]",
        r"\centering",
        r"\scriptsize",
        r"\setlength{\tabcolsep}{4pt}",
        (
            r"\caption{Width of a 95\% percentile interval for the scene mean, in points. "
            r"The observed column is the committed interval. "
            r"The $n{=}20$ and $n{=}50$ columns are extrapolations from the same scene list, "
            r"not measurements on more scenes. Full precision is in "
            r"\texttt{revision/analysis/sample\_size\_results.json}.}"
        ),
        r"\label{tab:s_power}",
        r"\begin{tabular}{lrrrr}",
        r"\toprule",
        r"Stored gaps & Observed $n$ & Stored width & $n{=}20$, extrapolation & $n{=}50$, extrapolation \\",
        r"\midrule",
    ]
    for row in rows:
        lines.append(
            f"{row['label']} & {row['n_observed']} & {one_decimal(row['stored_width'])} "
            f"& {one_decimal(row['hypothetical']['20']['width'])} "
            f"& {one_decimal(row['hypothetical']['50']['width'])} \\\\"
        )
    lines += [
        r"\bottomrule",
        r"\end{tabular}",
        r"\end{table}",
        "",
        (
            "Where the hypothetical $n$ is larger than the number of stored scenes, the width falls. "
            "For the hard paste, $n{=}20$ is smaller than the 32 stored scenes, so that width rises, "
            "and $n{=}50$ is where it falls. "
            "Neither column says what a scene outside the stored list would score."
        ),
        "",
    ]
    OUT_TEX.write_text("\n".join(lines))


def main() -> None:
    kill = load("kill_test_results.json")["test_val"]
    water = load("method1_results.json")["terrain"]
    ndvi = load("ndvi_comatch_results.json")["ndvi_comatch"]
    paste_rows = load("p2_results.json")["rows"]
    paste_ci = load("p2_analyze.out.json")["headline_val_test"]["ci95"]
    rows = [
        family_row(
            "Unmatched",
            [r["gap"] for r in kill["per_scene"]],
            0,
            kill["summary"]["ci95"],
        ),
        family_row(
            "Water fraction",
            [r["gap"] for r in water["matched"]],
            0,
            water["matched_ci95"],
        ),
        family_row(
            "NDVI co-match",
            [r["gap"] for r in ndvi["per_scene"]],
            0,
            ndvi["ci95"],
        ),
        family_row(
            "Hard paste",
            [r["gap_pts"] for r in paste_rows],
            42,
            paste_ci,
        ),
    ]
    info = inventory()
    if info["method1_scenes"] != 34 or info["paste_scenes"] != 32:
        raise SystemExit(f"unexpected inventory: {info}")
    if [r["n_observed"] for r in rows] != [8, 7, 5, 32]:
        raise SystemExit(f"unexpected scene counts: {[r['n_observed'] for r in rows]}")
    payload = {
        "draws": N_DRAWS,
        "hypothetical_n": list(HYPOTHETICAL),
        "note": (
            "n=20 and n=50 resample the committed per-scene gaps with replacement. "
            "They are extrapolations, not measurements on more scenes, and they are not point estimates."
        ),
        "inventory": info,
        "families": rows,
    }
    OUT_JSON.write_text(json.dumps(payload, indent=2) + "\n")
    write_tex(info, rows)
    print(f"wrote {OUT_JSON}")
    print(f"wrote {OUT_TEX}")
    for row in rows:
        print(
            row["label"],
            "n",
            row["n_observed"],
            "stored",
            one_decimal(row["stored_width"]),
            "n20",
            one_decimal(row["hypothetical"]["20"]["width"]),
            "n50",
            one_decimal(row["hypothetical"]["50"]["width"]),
        )


if __name__ == "__main__":
    main()
