# P2 same-tile cloud paint: locked rule (committed BEFORE any P2 number is computed)

Source: research/worldfloods-claim-2026-11-06/researchy-pass-01.md (Claim A, confirmatory arm P2).

- **Model**: released ml4floods `WF2_unetv2_all` (/workspace/wf-models/model.pt), frozen, zero-shot, water = sigmoid(ch1) > 0.5.
- **Tiles**: same grid as kill_test / cloud_iou_pilot (256x256, stride 512, tile kept if >=50% labelled water-band pixels).
- **Clear tile**: GT cloud fraction < 0.01 of labelled pixels. **Cloudy (mask-donor) tile**: GT cloud fraction >= 0.1.
- **Scenes**: every val+test scene with >= 200 GT-clear & GT-water pixels pooled over its clear tiles. No other filter.
- **Mask**: clear tile i of a scene receives the GT-cloud mask (gt band 0 == 2) of donor tile i mod k from the same scene's
  k cloudy tiles (ordered by row, col). If the scene has no cloudy tile, donors are all cloudy tiles of the same split with
  cloud fraction in [0.1, 0.6], ordered by (scene, row, col), tile i takes donor (hash-free) index (i + 42) mod K.
- **Spectrum**: per-band median of raw S2 bands 1-13 over GT-cloud pixels of the same scene if that scene has >= 1000
  GT-cloud pixels on its kept tiles; otherwise the per-band median over all GT-cloud pixels of kept val tiles (frozen once,
  applied unchanged to test). Painted pixels are set to this constant spectrum; then normalised as usual.
- **Scores (per scene, confusion pooled over its clear tiles)**:
  (i) HEADLINE: water IoU on unpainted labelled pixels, original vs after paint; gap = (orig - painted) x 100 points.
  (ii) secondary: water IoU after paint on painted labelled pixels vs original water labels.
- **Aggregate**: unweighted scene mean of gap; 95% CI from 5000-draw scene bootstrap, numpy default_rng(42).
- **Verdict on Claim A**: KILL if mean gap >= 5 and CI excludes 0. SUPPORT if CI upper < 5. Otherwise INCONCLUSIVE
  (incl. CI covering both 0 and 5). Reported per split as secondary, val+test pooled is the headline.
- No shuffled-mask control in this run (needed for Claim B only); no tuning or re-run with other settings.
