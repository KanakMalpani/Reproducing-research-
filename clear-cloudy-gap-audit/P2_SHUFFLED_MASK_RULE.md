# P2 follow-up: shuffled-mask control (locked BEFORE any number is computed)

Status: FOLLOW-UP / CONTROL. Does not change the locked confirmatory P2 verdict (KILL Claim A, 5.6 [3.0, 8.5]).
Motivation: Claim B hygiene (research/worldfloods-claim-2026-11-06/researchy-pass-01.md).

- Identical to P2_CLOUD_PAINT_RULE.md (model, tiles, scenes, donor assignment, spectrum, scoring, bootstrap) except the mask.
- **Shuffled mask**: donor mask cut into 16x16 px blocks (16x16 grid on the 256 tile); blocks randomly permuted with
  numpy default_rng(42), one generator consumed in P2's scene/tile order. Same painted pixel count; blob shape and
  location destroyed.
- The same pass re-computes real-mask P2 (sanity: must reproduce 5.6 within rounding).
- **Headline**: scene-mean drop in water IoU on UNPAINTED labelled pixels (shuffled), 5000-draw scene bootstrap rng 42,
  val+test pooled; plus paired difference real - shuffled, same bootstrap.
- **Interpretation, fixed in advance**:
  - paired real - shuffled CI includes 0 -> consistent with paste / distribution shift, not specific to a coherent cloud.
  - shuffled mean < 0.5 x real AND paired CI excludes 0 (real > shuffled) -> coherent cloud location matters (spillover supported).
  - shuffled > real, paired CI excludes 0 -> more cloud boundary = more damage (adjacency/boundary effect).
  - Otherwise inconclusive. Caveat recorded now: scattering lengthens the boundary, so unpainted pixels are on average
    closer to paint; P2_EDGE_DISTANCE_RULE.md is the cleaner adjacency test.
- Single run, no tuning.
