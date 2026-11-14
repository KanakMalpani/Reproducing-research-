# P2 follow-up: spatially varying spectrum paste (locked BEFORE any number is computed)

Status: FOLLOW-UP / HONESTY CHECK on the residual boundary/adjacency claim, after soft-edge (4.4 [2.3, 6.8], INCONCLUSIVE).
Does **not** change the locked P2 verdict (KILL Claim A, 5.6 [3.0, 8.5]); no NDVI re-run.
Motivation: soft-edge still used a spatially *constant* scene-mean fill. Here every painted pixel gets its own
spectrum drawn from real cloud pixels, so "non-constant" means per-pixel texture, not a flat fill.

## Protocol (reuse P2 / soft-edge; only spectrum differs)
- Model, tiles, clear/cloudy split, 32-scene list, donor-mask assignment (own same-scene mask, else borrowed
  pool index `(i+42) mod K`), scoring surface: identical to `P2_CLOUD_PAINT_RULE.md` / `P2_SOFT_EDGE_RULE.md`.
- **Cloud-pixel pool**: per scene, the same sampled GT-cloud pixels used for the P2 median / soft-edge mean
  (raw S2 bands 1–13, up to ~2000 px per tile with cloud>0 on kept tiles).
- **Per-pixel sampling**: for each tile, draw an index per pixel of the 256×256 grid uniformly **with replacement**
  from the pool; field `F[:, y, x] = pool[:, idx[y, x]]`. RNG: `numpy.default_rng(42)` created once before the
  loop, consumed in fixed scene/tile order. No spatial smoothing of F.
- **Fallback (tiny pool)**: if the scene has `ncloud < 1000` GT-cloud pixels (same threshold as soft-edge),
  sample from the **frozen val pool** = concatenation of all val-scene pools (computed once in the scan pass,
  applied unchanged to test).
- **Variants, both in one pass (cheap: one extra forward each)**:
  - **Headline — soft + varying**: `A = gaussian_filter(M.astype(float32), sigma=2.0, truncate=3.0)`;
    `x' = (1 − A)·x + A·F`.
  - **Secondary — hard + varying**: `x'[:, M] = F[:, M]`.
  - Sanity: hard constant-median P2 recomputed (should reproduce ~5.6).
- **Primary metric**: visible-pixel water IoU drop on `v & ~M` (hard mask), (orig − painted) × 100; scene-mean
  over the 32 scenes; 95% CI from 5000-draw scene bootstrap, `numpy.default_rng(42)`.
- **Edge bins** (`P2_EDGE_DISTANCE_RULE.md`): `distance_transform_edt(~M)`, bins (0,8], (8,32], >32 px, for both
  varying variants.

## Pre-declared interpretation (headline = soft + varying)
- CI lower ≤ 0 → further soften residual claim in open-risks (and, only if honesty requires, a one-line residual
  soften in the pasteable body, quoted to Eng Lead; never marked Kanak-approved).
- Mean ≥ 5 AND CI lower > 0 (and (0,8] bin ≫ >32 bin) → residual strengthens under more realistic paste.
- Otherwise → INCONCLUSIVE; keep current tempered wording; record in open-risks.
- Hard + varying reported as secondary using the same bars; does not override the headline.
- Compare to hard 5.6 [3.0, 8.5], soft 4.4 [2.3, 6.8], shuffled 8.1 [4.1, 12.8].

## Out of scope
No σ / pool / threshold tuning; single CPU run; no edits to preregistration.md, RS abstracts, or pasteable abstract body
(except the honesty case above).
