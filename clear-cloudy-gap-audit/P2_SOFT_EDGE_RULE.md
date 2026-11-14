# P2 follow-up: soft-edge / own-scene-mean spectrum paste (locked BEFORE any number is computed)

Status: FOLLOW-UP / HONESTY CHECK on the residual boundary/adjacency claim.
Does **not** change the locked confirmatory P2 verdict (KILL Claim A, 5.6 [3.0, 8.5]), and does **not** re-run NDVI.
Motivation: hard-edged constant-spectrum paste (P2 + shuffled) leaves a residual concentrated within ~8 px of the paint
edge. A soft feather + scene-mean spectrum tests whether that residual survives more natural boundaries. Preferred
evidence for the residual claim before the 16 Nov paper; registration abstract already tempers to synthetic
boundary/adjacency under hard-edged paste (`7374a2c`).

## Protocol (reuse P2; only soft edge + spectrum differ)
- Identical to `P2_CLOUD_PAINT_RULE.md` for model, tiles, scenes, donor assignment, scoring surface, bootstrap:
  - Model: frozen ml4floods `WF2_unetv2_all`, water = sigmoid(ch1) > 0.5.
  - Tiles / clear / cloudy / scenes: same as P2 (`p2_cloud_paint.py` / `p2_shuffled_control.py`).
  - Donor mask: same same-scene cloudy-tile mask (or borrowed pool with index `(i+42) mod K`).
  - **Hard mask M** (boolean) is the donor GT-cloud mask; it defines "painted" vs "unpainted" for scoring.
- **Spectrum (non-constant across scenes; spatially constant within a paste)**:
  - If the scene has `ncloud >= 1000` GT-cloud pixels on kept tiles: use that scene's **per-band mean** of raw S2
    bands 1–13 over sampled GT-cloud pixels (same sampling as P2: up to ~2000 px/tile). This replaces P2's scene
    **median** for the soft-edge honesty check.
  - Else: same fallback as P2 — frozen val-split spectrum (median of per-scene val cloud medians, computed once
    from the scan pass and applied unchanged to test).
  - Own-cloud / borrowed **mask** policy unchanged from P2.
- **Soft edge (exact kernel, locked here)**:
  - Let `M` be the hard boolean donor mask on the 256×256 tile.
  - Soft alpha `A = gaussian_filter(M.astype(float32), sigma=2.0, truncate=3.0)` via `scipy.ndimage.gaussian_filter`
    (isotropic; default mode/cval). Effective feather width ≈ 6 px (≈3σ); inside the locked 4–8 px range.
  - Paste: `x_soft = (1 − A) * x + A * spec[:, None, None]` (broadcast over H×W; A clipped implicitly by the blur
    into [0, 1]).
  - No other blur, morphological open/close, or distance ramp. Single setting; no tuning.
- **Primary metric**: water IoU drop on **unpainted labelled pixels** = `v & ~M` (hard mask), original vs after soft
  paint; gap = (orig − painted) × 100 points. Same as P2 headline. Scene-mean over the same 32-scene list; 95% CI
  from 5000-draw scene bootstrap, `numpy.default_rng(42)`. Seed fixed before the run.
- **Edge-distance bins** (reuse `P2_EDGE_DISTANCE_RULE.md`): Euclidean distance from each unpainted pixel to nearest
  hard-painted pixel (`scipy.ndimage.distance_transform_edt(~M)`); bins (0,8], (8,32], >32 px. Per-scene pooled
  confusion per bin; scene-mean gap + same bootstrap over scenes with finite IoU in that bin. Computed on the
  soft-edge paste (not a re-run of hard P2).
- Also recompute hard-edge P2 in the same pass as a sanity check (must reproduce ~5.6 within rounding); soft is the
  headline of this follow-up.

## Pre-declared interpretation (vs hard P2 5.6 and shuffled 8.1)
- If soft-edge mean drop has **CI lower ≤ 0**: soften the residual boundary/adjacency claim further (hard-edge
  artefact; abstract residual wording must be weakened).
- If soft-edge mean drop **stays ≥ 5 with CI lower > 0** AND the drop still concentrates in the (0,8] bin (near-edge
  ≫ far bins): residual boundary claim **strengthens** under more natural edges.
- Otherwise: inconclusive honesty check; keep current tempered wording (synthetic boundary/adjacency under
  hard-edged paste) and note soft-edge as unresolved.
- Compare numerically to hard P2 5.6 [3.0, 8.5] and shuffled 8.1 [4.1, 12.8]; do not re-litigate Claim A.

## Out of scope
- No NDVI re-run; no change to Claim A kill; no edits to `preregistration.md` or RS abstracts.
- Single CPU run; no hyperparameter search on σ or spectrum estimator.
