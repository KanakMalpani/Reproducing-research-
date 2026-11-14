# NDVI co-match (locked BEFORE any number is computed)

Status: PRE-SPECIFIED MATCHING ARM (Feynman ff1d44b / NOTE.md). Does not re-run P2 or change the Claim A kill (5.6 [3.0, 8.5]).
Motivation: water-fraction-only match gave 8.85 [-0.66, 17.78] on 7 scenes (INCONCLUSIVE; CI lower ≤ 0). Register a matched cloud-gap claim only if this stricter co-match also clears the bar; otherwise drop topic framing that relies on a matched cloud gap (keep P2 residual / composition paper).

## Matching (within each paired scene)
- Clear tiles: `cloud_frac < 0.01`. Cloudy tiles: `cloud_frac >= 0.1`.
- NDVI field: `ndvi_clear` in `method1_tiles.json` (already computed). Skip any tile with `ndvi_clear is None` when matching on NDVI.
- For each cloudy tile (sorted by water_frac, same order as water-only match in `method1_analyze.py`), assign the nearest unused clear tile that satisfies BOTH:
  - `|Δ water_frac| ≤ 0.05`
  - `|Δ NDVI| ≤ 0.05` where Δ NDVI = `|ndvi_clear(cloudy) − ndvi_clear(clear)|`
- "Nearest" = minimum `|Δ water_frac|` among clear tiles that meet both tolerances (ties broken by enumerate index, as in method1).
- Both matched pools must have ≥200 water pixels: `pa[0]+pa[2] >= 200` and `pb[0]+pb[2] >= 200`, using confusion matrix `cm["base"]` (same as water-only).

## Gap and aggregate
- Per scene: `gap = 100 * (IoU_clear − IoU_cloudy)` on the matched pools (`IoU = cm[0] / max(sum(cm), 1)` on pooled `cm["base"]`).
- Aggregate: mean of per-scene gaps.
- Bootstrap: 5000-draw scene bootstrap; **seed fixed first** with `np.random.default_rng(0)` (same as `method1_analyze.py`).
- Report: per-scene pairs count and gaps, mean, 95% CI, n_scenes, and how many scenes survive vs water-only match (currently 7).

## Verdict (fixed in advance)
- **PASS / register matched claim** only if mean ≥ 5.0 AND CI lower bound > 0.
- If CI lower ≤ 0: **DROP** topic framing that relies on a matched cloud gap; still keep the P2 residual / composition paper. Verdict string: DROP (or INCONCLUSIVE if mean < 5 but CI lower > 0 — still do not register the matched claim).
- Do NOT re-run P2. Do NOT change the Claim A kill.

## Inputs / outputs
- Input: `topics/worldfloods-cloud-aware/method1_tiles.json`.
- Script: `ndvi_comatch_analyze.py` (extends water-only logic in `method1_analyze.py`).
- Output: `ndvi_comatch_results.json`.
- Single run, no tuning after seeing the number.
