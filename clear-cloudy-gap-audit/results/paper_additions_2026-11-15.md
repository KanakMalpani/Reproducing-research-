# Paper additions — 2026-11-15

The user confirmed that a PDF is not a required input. These additions have been inserted beside the corresponding results in the available audit narrative, `NOTE.md`. A LaTeX fragment accompanies this file. The separately referenced canonical `paper/main.tex` and `paper/PAPER_DRAFT.md` are absent, so no edit to those unavailable files is claimed. Values below are copied from existing result JSON; no model was trained and no result JSON was replaced.

## Beside the existing hard-paste paragraph

Keep the printed hard-paste result **5.6 [3.0, 8.5], n=32**, and the printed test scene mean **3.8 [1.5, 6.9]** unchanged. Insert beside them:

As a separate corrected pooled-median check on frozen WF2 (seed 42), using identical masks and constant fills on clear tiles and scoring only unpainted valid pixels, the 32-scene cloud-fill drop is 5.401824562730654 points [3.0395970810675137, 8.131108851983626], the land-fill (non-cloud placebo) drop is 1.875462012891159 [0.7947312280281873, 3.0859464397899936], and the paired cloud-minus-placebo contrast is 3.5263625498394946 [1.4732143046767636, 5.72402744697899]; on 17 test scenes these are respectively 4.444186880094727 [1.6715664311563256, 8.530741356636542], 2.3272143974326833 [0.7203564862910251, 4.1754992673787115], and 2.1169724826620437 [-0.3760086187473995, 5.187757191515272], whose paired interval includes zero; all intervals in this check use scene bootstraps, and both fills are constructed fills rather than natural cloud.

Evidence: `pooled_cloud_noncloud_placebo.json`, `summary.val_test` and `summary.test`. These corrected-rule values do not replace either printed paste estimate. They are also distinct from the original event interval [3.1, 8.5] and event-weighted 6.0; neither original quantity is recomputed here.

## Beside the existing NDVI/composition result

The printed NDVI-matched 1.4-point gap belongs to the released UNet; frozen Prithvi on the same five scenes instead has an NDVI-matched gap of 26.413202413471293 points [-2.479672216469317, 56.2720287818474], and its 32-scene cloud-fill drop is 1.48590782566094 [0.4696010095725278, 2.557471004375826], compared with a larger non-cloud-placebo drop of 2.6482709365569335 [1.8216041771839329, 3.552474710682945] and a paired cloud-minus-placebo contrast of -1.1623631108959933 [-2.2004243456862635, -0.15842118822902607]; the smaller cloud-fill drop therefore does not establish a better model, because its placebo drop is larger.

Evidence: `prithvi_ndvi_comatch_results.json`, `ndvi_comatch`; `prithvi_same_protocol.json`, `summary.val_test`. These are not the frozen UNet's 39.3 or 1.4. The candidate's unmatched 34.59240223938893 and NDVI-matched 4.294778843697456 remain separate as well.

## Brightness census paragraph

For brightness defined as sqrt(B2^2+B3^2+B4^2)/10000, at a cut of 0.3 the GT-cloud counts at or below the cut are 1413748 pixels (237524 water), versus 710643 above (232857 water), and the GT-clear counts are 71202656 at or below (18076676 water), versus 4959997 above (2489855 water); at a cut of 0.35 the corresponding GT-cloud counts are 1951054 at or below (434810 water), versus 173337 above (35571 water), and the GT-clear counts are 74323427 at or below (19498563 water), versus 1839226 above (1067968 water); these counts are a brightness census, not a thin-cloud class.

Evidence: `brightness_census.json`, `total`. Each bin reports all valid pixels, then GT-water pixels in that bin.

## Verified separation checks; retain when inserting the LaTeX fragment

- Preserve every previously printed frozen number and its aggregator.
- Keep corrected-rule scene-bootstrap intervals distinct from the original printed paste and original event-clustered quantities.
- Keep Prithvi and the trained candidate distinct from the frozen UNet results.
- Keep Campo Galliano (IoU 0.19791666666666666) and Ituango South (IoU 0.7203674344866176) outside both gap samples n=8 and n=32; neither has a stored clear–cloudy gap.
- Keep permanent-water scene-mean IoU 0.9975072173600821 labelled as a label-dependent stratum score, never a clear–cloudy gap.
- The full genuine permanent-versus-flood split remains blocked; JRC code 0 is never labelled flood.
