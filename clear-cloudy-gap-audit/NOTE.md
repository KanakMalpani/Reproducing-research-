# WorldFloods v2 cloud-aware pilot (CPU, 2026-11-03)

Data: WorldFloods v2 (HF `isp-uv-es/WorldFloodsv2`, CC BY-NC), local 40 GB subset at `/home/box/worldfloods` (see `data/DATASETS.md`); not in git.
Baseline: published `isp-uv-es/ml4floods` `WF2_unetv2_all` (13-band UNet, 31 MB), unchanged, threshold 0.5.
Protocol: test+val scenes (34 with readable S2), 256x256 tiles on a 512-stride grid, keep tiles with >=50% valid water labels (1291 tiles).
Cloud fraction = share of valid pixels with GT cloud = 2. Water IoU is pooled per bin; 95% CI from a scene-level bootstrap (500 draws).
Run: `/workspace/wf-venv/bin/python cloud_iou_pilot.py test val`.

| cloud bin | tiles | scenes | water IoU (all px) | 95% CI | IoU clear px | IoU under-cloud px |
|---|---|---|---|---|---|---|
| [0, 0.01) | 992 | 34 | 0.828 | 0.772–0.902 | 0.828 | n/a (few px) |
| [0.01, 0.1) | 173 | 13 | 0.760 | 0.537–0.879 | 0.759 | 0.783 |
| [0.1, 0.3) | 91 | 9 | 0.730 | 0.602–0.779 | 0.738 | 0.680 |
| [0.3, 0.6) | 27 | 4 | 0.802 | 0.584–0.859 | 0.816 | 0.783 |
| [0.6, 1) | 8 | 2 | 0.684 | 0.503–0.745 | 0.661 | 0.694 |

Tile-level Spearman(cloud fraction, IoU) = -0.06 (p = 0.11).

Reading: IoU drops about 7–15 points from clear tiles to cloudy bins, but CIs overlap, the cloudy bins hold only 2–13 scenes, and the trend is not monotone. Cloud level is confounded with the scene. On this evidence the degradation is real but modest and statistically weak.

What a cloud-aware method would add: the published model already outputs a cloud head, but its water head is unchanged under thin cloud and cloud shadow, and pixels under thick cloud are just guessed. A cloud-aware method, such as cloud-probability-conditioned water decoding or test-time adaptation weighted by the cloud head, would aim to recover water IoU in the cloudy bins without hurting clear tiles. Falsifying test (to fix before running): on held-out test scenes, the method must raise pooled water IoU in the cloud-fraction >= 0.1 bins by at least 5 points, with a scene-bootstrap 95% CI excluding 0, and must not lower clear-bin IoU by more than 1 point. If the baseline gap between clear and cloudy bins stays under 5 points once controlled per scene (same-scene clear vs cloudy tiles), the topic has too little headroom and should be dropped.

## Val pixel census (2026-11-03, `val_cloud_counts.py`, `val_cloud_counts.json`)
16 val scenes. The GT cloud mask is a thresholded s2cloudless map, so "thin" had to be redefined after a first pass found 0 thin pixels among GT-cloud pixels (this change was made after looking at the data).
v2 definitions: clear = GT-clear and s2cloudless < 20; thin = GT-clear and s2cloudless >= 20; thick = GT-cloud.
Labelled px: clear 122.2M, thin 27.9M, thick 3.05M. Flood px: clear 35.4M, thin 0.65M, thick 0.25M (0.7% of flood).
Thick cloud appears in 8/16 scenes; only Falun (2.18M px) and Toamasina (0.68M) have much.

## Kill test: same-scene clear vs cloudy tiles (`kill_test.py`, `kill_test_results.json`)
Per scene: pooled water IoU of clear tiles (cloud fraction < 0.01) minus cloudy tiles (>= 0.1). A scene counts only if both groups have at least 200 water px. 95% CI from a 5000-draw scene bootstrap. Zero-shot only (val + test); train is excluded because the checkpoint was trained on it.
- val+test: 8 scenes, mean gap 39.3 pts (median 36.4), CI 22.4–57.6; clear tiles do better in 8/8 scenes. **Honesty (Feynman ff1d44b): do NOT treat this unmatched 39.3 as PASS for a cloud effect.** The gap tracks water fraction and terrain (see Terrain check); it is a raw clear-vs-cloudy composition gap, not a causal cloud claim. Leading abstracts with 39.3 as evidence of cloud failure is dead.
- val alone: 3 scenes, 47.3 pts, CI 19.6–85.8. test alone: 5 scenes, 34.5 pts, CI 14.6–54.4.
- Scoring cloudy tiles on visible (GT-clear) pixels only: 32.7 pts, CI 15.6–52.4. Same caveat: unmatched, composition-confounded.
- The pooled-bin table above understated the gap because the scenes with the most cloud also have easier water (a Simpson-type mix).
Caveats: only 8 paired scenes; cloudy tiles may sit in different land cover than clear tiles within the same scene; thick-cloud flood is only 0.7% of flood px.

### Matched-claim kill criteria (Feynman; register only if both hold)
Pass a water-fraction-matched cloud-gap claim only if: mean gap ≥5.0 pts AND 95% scene-bootstrap CI lower bound >0 (|Δ water fraction|≤0.05 within scene; both groups ≥200 water px). Present matched gap **8.9 [-0.7, 17.8] is INCONCLUSIVE** (lower bound −0.7 ≤ 0). Drop matched-cloud-gap framing if a pre-specified NDVI co-match (|Δ water|≤0.05 AND |Δ NDVI|≤0.05) also has lower bound ≤0 (still keep P2 residual / composition paper).
**NDVI co-match RAN (rule `NDVI_COMATCH_RULE.md` locked at 0df216c before numbers):** 5 scenes (2 of 7 water-only scenes dropped: EMSR286, EMSR438), mean gap **1.4 pts, 95% CI −24.5 to 24.5** (seed 0, 5000-draw). Verdict **DROP** matched-cloud-gap framing (CI lower ≤ 0; mean ≪ 5). Do not register a matched cloud-gap claim. Paper stays composition + small P2 residual.

**Frozen cross-model comparison (separate follow-up).** The printed NDVI-matched 1.4-point gap belongs to the released UNet; frozen Prithvi on the same five scenes instead has an NDVI-matched gap of 26.413202413471293 points [-2.479672216469317, 56.2720287818474], and its 32-scene cloud-fill drop is 1.48590782566094 [0.4696010095725278, 2.557471004375826], compared with a larger non-cloud-placebo drop of 2.6482709365569335 [1.8216041771839329, 3.552474710682945] and a paired cloud-minus-placebo contrast of -1.1623631108959933 [-2.2004243456862635, -0.15842118822902607]; the smaller cloud-fill drop therefore does not establish a better model, because its placebo drop is larger.


## Terrain check (`method1_pass.py`, `method1_analyze.py`, `method1_results.json`)
In the 8 paired scenes, clear tiles are usually water-rich permanent water (for example Toamasina: water fraction 0.85 clear vs 0.07 cloudy; JRC permanent water 0.84 vs 0.04). Cloudy tiles are mostly vegetated land with little water (NDVI higher on the cloudy side in 5/8 scenes: Toamasina, Candon, Vatomandry, Ituango North, AOI02; clear side higher in Falun, North Normanton, Mwanza). An earlier sentence of this note said 6/8. The tile file (`method1_results.json`) says 5/8. The enrolled abstract stays qualitative and was not edited. JRC permanent water is higher on the clear tiles in 7/8; North Normanton is the exception, and Mwanza is ~0 on both sides.
The tile pairs matched on water fraction (each cloudy tile paired with the nearest unused clear tile within 0.05, same scene) cover 7 scenes. Their mean gap is **8.9 pts, 95% CI −0.7 to 17.8**, down from 39.3. Most of the raw gap is terrain and water fraction, not cloud. After matching, the 5-pt-bar verdict is **INCONCLUSIVE** (the CI straddles 5 and includes 0).

## NDVI co-match (`NDVI_COMATCH_RULE.md`, `ndvi_comatch_analyze.py`, `ndvi_comatch_results.json`)
Rule committed at 0df216c before running. Same within-scene greedy match as water-only, plus `|Δ ndvi_clear| ≤ 0.05`; skip tiles with `ndvi_clear is None`; both pools ≥200 water px on `cm["base"]`; gap = 100·(IoU_clear−IoU_cloudy); mean of per-scene gaps; 5000-draw scene bootstrap with `np.random.default_rng(0)`.
- Surviving scenes: **5** (water-only had 7; dropped EMSR286_08ITUANGONORTH, EMSR438_AOI02).
- Mean gap **1.43 pts**, 95% CI **[−24.53, 24.51]**. Per-scene gaps: Falun 7.3, Candon 39.0, Vatomandry −8.2, NorthNormanton 14.9, Mwanza −45.7.
- Locked verdict: **DROP** matched-cloud-gap framing (mean < 5 and CI lower ≤ 0). Claim A kill from P2 unchanged. Abstract must not promise a matched cloud-gap claim.

## Method 1: cloud-adaptive threshold / cloud-head gate (rule in `SELECTION_RULE_method1.md`, committed before running)
Val selection picked T+0.4 (threshold 0.5 + 0.4·cp). Val cloudy-tile IoU went from 67.5 to 72.2, and clear tiles rose by 0.7. The pick sits at the edge of the grid; this is noted but the grid stays as committed.
Test, scored once: cloudy-tile IoU delta **−0.24 pts** (95% CI −1.4 to +10.4, 6 scenes); clear tiles +0.21. **FAIL** against the bar (needs +5 with the CI excluding 0).

## Method 2: cloud-aware output adapter (rule in `SELECTION_RULE_method2.md`; `m2_results.json`)
Frozen WF2 UNet. A residual adapter was trained on 7653 train tiles (4 epochs, CPU). Val baseline: clear 86.7, cloudy 67.5.
Val (clear / cloudy): h8_cp1 91.0/76.1, h8_cp0 91.0/76.2, h16_cp1 91.0/76.2, h16_cp0 (see json). Selected: h16_cp1.
The variants without the cloud-probability input do just as well, so the val gain is general recalibration, not cloud awareness.
Test, scored once: cloudy-tile IoU 76.7 to 77.1 (**+0.44 pts**, CI −2.3 to +15.9); clear +0.55. **FAIL** against the bar.
Terrain-matched on test (5 scenes): baseline gap 10.2 (CI −0.4 to 19.8) and matched cloudy IoU 75.3; selected gap 13.9 (CI 3.3 to 27.1) and matched cloudy IoU 77.1.
Exploratory, val only (wider threshold grid a in {0.5..1.5}): cloudy IoU 72.5 to 73.4 and clear 87.5 to 88.7. It keeps rising at the edge again; not scored on test.
Overall: neither cheap method closes the cloud gap. WorldFloods cloud-aware evidence is weak going into Thu 5 Nov.

## P2: same-tile cloud paint (confirmatory, rule `P2_CLOUD_PAINT_RULE.md` committed at 2f89651 before running)
32 val+test scenes with >=200 clear-water px (15 val, 17 test). GT-cloud masks transplanted onto clear tiles, filled with a
constant thick-cloud spectrum; frozen UNet, threshold 0.5. Results in `p2_results.json`, `p2_analyze.out.json`.
- **Headline (val+test): drop in water IoU on UNPAINTED pixels = 5.6 pts, 95% CI [3.0, 8.5]** (median 2.3; 24/32 scenes drop).
  Locked verdict: **KILL Claim A** (mean >= 5 and CI excludes 0). Val 7.6 [3.0, 12.4]; test 3.8 [1.5, 6.9] (INCONCLUSIVE alone).

**Corrected-rule placebo check (separate follow-up).** As a separate corrected pooled-median check on frozen WF2 (seed 42), using identical masks and constant fills on clear tiles and scoring only unpainted valid pixels, the 32-scene cloud-fill drop is 5.401824562730654 points [3.0395970810675137, 8.131108851983626], the land-fill (non-cloud placebo) drop is 1.875462012891159 [0.7947312280281873, 3.0859464397899936], and the paired cloud-minus-placebo contrast is 3.5263625498394946 [1.4732143046767636, 5.72402744697899]; on 17 test scenes these are respectively 4.444186880094727 [1.6715664311563256, 8.530741356636542], 2.3272143974326833 [0.7203564862910251, 4.1754992673787115], and 2.1169724826620437 [-0.3760086187473995, 5.187757191515272], whose paired interval includes zero; all intervals in this check use scene bootstraps, and both fills are constructed fills rather than natural cloud.

- Secondary: water IoU on painted pixels vs original labels: scene-mean 0.38 (unpainted original 0.78). Thick cloud hides water, as expected.
- Implementation deviation (logged, not a re-run): the frozen val spectrum was the median of per-scene val medians, not the
  pooled median of all val GT-cloud pixels.
- **Exploratory (post hoc, not a verdict):** the drop is concentrated where the rule's fallbacks fired. Scenes with their own
  cloudy-tile mask AND own spectrum (n=8): 1.5 [0.3, 2.9]. Scenes with a borrowed mask or frozen spectrum (n=24): 7.0 [3.7, 10.5];
  frozen spectrum only (n=19): 8.6 [4.5, 12.8]. So the kill may partly be an out-of-scene paste artefact.
- Reading: with terrain fixed, a pasted thick cloud degrades water calls in visible pixels next to it by about 5 pts on
  average. The "gap is largely a confound" sentence cannot be registered. Of the raw 39.3 pts, the matched 8.9 and the P2
  5.6 are consistent with a synthetic boundary/adjacency sensitivity under hard-edged paste plus a large composition effect
  — not established natural-cloud causality. Shuffled-mask and soft-edge honesty checks completed (see follow-ups);
  soft-edge 4.4 [2.3, 6.8] is inconclusive vs the ≥5 bar (see soft-edge section); spatially varying spectrum paste
  1.7 [0.5, 2.7] is also inconclusive but shows most of the 5.6 depends on a flat constant fill (see varying section).

## P2 follow-ups (labelled follow-up / sensitivity; the confirmatory P2 KILL of Claim A is unchanged)
Rules committed before running at 195f441: `P2_SHUFFLED_MASK_RULE.md`, `P2_OWN_CLOUD_ONLY_RULE.md`, `P2_EDGE_DISTANCE_RULE.md`.
Code `p2_shuffled_control.py`, `p2_followup_analyze.py`; outputs `p2_shuffled_results.json`, `p2_followup_analyze.out.json`.
- Same-pass re-computation of real-mask P2 reproduces 5.59 [3.00, 8.51] (32 scenes).
- **Shuffled mask (16 px blocks, rng 42)**: 8.1 [4.1, 12.8]; paired real - shuffled = -2.5 [-4.5, -0.6]. Pre-locked reading:
  shuffled > real with CI excluding 0 = **boundary/adjacency effect** (more cloud edge, more damage); not a pure global paste shift.
- **Edge distance** (scene mean drop, pts): real mask 0-8 px 9.4 [5.4, 13.7], 8-32 px 0.2 [-0.4, 0.8], >32 px 0.0 [-0.03, 0.0] (n=32 each).
  Shuffled 9.1 [4.8, 14.0] (n=32) / 1.1 [-0.7, 3.2] (n=32) / 0.0 [-0.2, 0.1] (n=17). The parenthetical n=17 applies only to the shuffled >32 px bin (`p2_followup_analyze.out.json`). The visible-pixel drop is confined to within ~8 px of paint. The real >32 px JSON interval is [-0.026, -0.001] (mean -0.01 pt): a hundredth of a point, entirely on the side of a tiny gain. The NOTE rounding 0.0 [-0.03, 0.0] is the magnitude to quote.
- **Own-cloud-only stratum (post hoc, n=8)**: 1.5 [0.3, 2.9]; complement (borrowed mask or frozen spectrum, n=24) 7.0 [3.7, 10.5].
  Shuffled own-cloud-only 1.0 [-1.2, 3.2]. Descriptive only: the size of the effect depends on which spectrum/mask is pasted.
- Caveat: constant-spectrum hard-edged paste is unrealistic; a near-edge drop may partly reflect the sharp synthetic boundary.

## P2 soft-edge / own-mean spectrum (honesty check; rule `P2_SOFT_EDGE_RULE.md` locked at c612c1e before numbers)
Follow-up only; Claim A kill and NDVI DROP unchanged. Soft alpha = `gaussian_filter(M.astype(float32), sigma=2.0, truncate=3.0)`
(~6 px feather); spectrum = scene-mean when ncloud≥1000 else frozen val (same fallback as P2). Score on hard `v & ~M`.
Code `p2_soft_edge.py`, `p2_soft_edge_analyze.py`; outputs `p2_soft_edge_results.json`, `p2_soft_edge_analyze.out.json`.
- **Headline soft-edge drop**: **4.4 pts, 95% CI [2.3, 6.8]** (n=32). Hard-edge sanity in the same pass: 5.59 [3.00, 8.51]
  (reproduces confirmatory P2).
- **Edge bins (soft paste)**: 0–8 px **8.1 [4.9, 11.6]**; 8–32 px −0.07 [−0.74, 0.52]; >32 px −0.01 [−0.03, 0.00].
  Still confined to within ~8 px of the hard paint edge.
- Own-cloud-only soft (n=8, descriptive): 1.1 [−0.09, 2.3]; complement 5.5 [2.8, 8.6].
- Compare: hard P2 5.6 [3.0, 8.5]; shuffled 8.1 [4.1, 12.8]. Soft is between 0 and hard, closer to hard than to null.
- **Pre-declared verdict: INCONCLUSIVE honesty check** (mean 4.4 < 5, so neither "soften residual further" [needs CI
  lower ≤ 0] nor "strengthen" [needs mean ≥ 5 with lower > 0]). Residual boundary/adjacency still excludes 0 and is
  edge-local under soft paste, but does not clear the ≥5 bar. Keep current tempered abstract wording
  ("synthetic boundary/adjacency under hard-edged paste"); soft-edge does not force a body rewrite.

## P2 spatially varying spectrum (honesty check; rule `P2_VARYING_SPECTRUM_RULE.md` locked at 4758576 before numbers)
Follow-up only; Claim A kill / NDVI DROP unchanged. Each painted pixel samples (with replacement, `default_rng(42)`)
a raw S2 13-band spectrum from the scene's real GT-cloud pixel pool (same ≤~2000 px/tile sampling as P2); scenes with
ncloud<1000 use the frozen val pool (all val pools concatenated). Combined with the soft-edge feather (σ=2.0, truncate=3.0)
as headline; hard-edge varying as secondary; constant-median hard P2 recomputed as sanity. Score on hard `v & ~M`.
Code `p2_varying_spectrum.py`, `p2_varying_analyze.py`; outputs `p2_varying_results.json`, `p2_varying_analyze.out.json`.
- **Headline soft + varying**: **1.7 pts [0.5, 2.7]** (n=32). Hard + varying: 1.9 [0.4, 3.4]. Sanity 5.59 [3.00, 8.51].
- **Edge bins (soft+varying)**: 0–8 px **4.2 [2.1, 6.2]**; 8–32 px −0.36 [−0.97, 0.26]; >32 px −0.02 [−0.05, −0.001].
  Hard+varying: 4.7 [2.0, 7.6] / −0.26 / −0.02. Still edge-local.
- Own-cloud-only (n=8, descriptive): 0.8 [−0.3, 2.1]; complement 1.9 [0.5, 3.2].
- Compare: hard 5.6 [3.0, 8.5]; soft 4.4 [2.3, 6.8]; shuffled 8.1 [4.1, 12.8].
- **Pre-declared verdict: INCONCLUSIVE** (CI lower > 0, mean < 5). Reading: texture matters more than the edge feather —
  roughly two-thirds of the hard 5.6 disappears when the paste carries real per-pixel cloud variability. A small
  (~1–2 pt) edge-local residual still excludes 0. The "~5.6" magnitude should be read as an upper bound tied to
  constant-spectrum paste. Pasteable body not changed (soften bar not hit); flagged in open-risks for Kanak.

## Test label histogram (rule `LABEL_HISTOGRAM_RULE.md`; `label_histogram_test.json`)
Test GT-cloud: 7.70M labelled px, **0.0% with s2cloudless < 50** (all in 50-100). GT-clear: 159.4M px, 6.9% with s2cloudless >= 20.
Confirms on test what val showed: the released cloud label is a hard s2cloudless cut with **no thin class inside GT-cloud**;
thin/hazy pixels are labelled clear. GT-cloud visible brightness (mean B2-B4) is mostly 500-2000 DN.

**Brightness census (separate follow-up).** For brightness defined as sqrt(B2^2+B3^2+B4^2)/10000, at a cut of 0.3 the GT-cloud counts at or below the cut are 1413748 pixels (237524 water), versus 710643 above (232857 water), and the GT-clear counts are 71202656 at or below (18076676 water), versus 4959997 above (2489855 water); at a cut of 0.35 the corresponding GT-cloud counts are 1951054 at or below (434810 water), versus 173337 above (35571 water), and the GT-clear counts are 74323427 at or below (19498563 water), versus 1839226 above (1067968 water); these counts are a brightness census, not a thin-cloud class.


## Paper pass (2026-11-06, falsifiable core)
No new run. The CVPR draft now states what the unmatched gap measures (scene-mean water IoU, clear versus cloudy under the released label) and what it does not (natural-cloud causality, a thin class, a pixel-pooled score). KILL / DROP / INCONCLUSIVE / FAIL are defined as statements about pre-registered bars. The Claim A kill remains the 32-scene hard-paste mean 5.6 [3.0, 8.5]. Summing the stored unpainted counts gives the already-reported 2.8 and 0.3 point estimates; they are shown beside the scene mean and left unscored. Soft-edge 4.4 and varying-spectrum 1.7 stay INCONCLUSIVE and out of the abstract. The enrolled OpenReview abstract was not edited.

## Paper draft (2026-11-06) and enrolled-text notes
Canonical CVPR draft: `paper/main.tex`, with the honesty ladder in `paper/figures/honesty_ladder.pdf` transcribed from the locked intervals (no new run). `paper/PAPER_DRAFT.md` is the same argument in markdown.

Enrolled pasteable body (`research/registration-2026-11-10/WORLDFLOODS_ABSTRACT.md`) was **not** edited. Two precision notes, recorded in that file's open-risks only:

- The enrolled clause "paints real cloud spectra" is the P2 constant per-band spectrum taken from real GT-cloud pixels (scene median, or the frozen val median-of-medians). It is not the per-pixel draw. That draw is the varying-spectrum check, 1.7 [0.5, 2.7], which stays out of the enrolled body by the pre-declared rule. The paper says "constant spectrum" explicitly. No numeric mismatch with 5.6 [3.0, 8.5].
- The enrolled composition sentence is qualitative ("permanent water" / "vegetated land"). It does not quote the old 6/8 NDVI count. North Normanton has the opposite water-fraction sign (cloudy 0.48 vs clear 0.31) and a 2.1 pt raw gap. Not edited.
