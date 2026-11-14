# Remaining steps

The user finishes this list. Items 1–3, 5, and 6 are already done. Evidence is in [REMAINING_STATUS.md](REMAINING_STATUS.md). Write every new JSON file under `clear-cloudy-gap-audit/results/`. Do not replace the JSON already in this folder. Do not invent IoU numbers, scene counts, or intervals. If a download or a path is missing, stop. Leave every printed frozen number unchanged.

## Do not train

Do not train another model. The six-epoch candidate missed its target (`target_met: false`). Test cloudy IoU fell from `0.7667548616804217` to `0.7390338550118456`.

A new checkpoint is not a paper improvement unless it is scored with the composition control, the same-tile control, and the aggregator. It does not replace the non-cloud paste, the permanent-versus-flood split, or the frozen Prithvi run.

The paste stays a constructed fill on clear tiles, scored on unpainted pixels. It is not natural cloud.

## Already done

- [x] On Kaggle or Colab, this folder is the only code. Set `WF_ROOT` and `WF_CKPT`. Download validation and test from `isp-uv-es/WorldFloodsv2`, and `model.pt` from `isp-uv-es/ml4floods` at `models/WF2_unetv2_all`.
- [x] Reproduce the JSON already in this folder before any new number: the kill test, the NDVI co-match, and the hard paste.
- [x] Non-cloud paste placebo, and a re-run of the frozen spectrum under the written pooled-median rule. Seed 42.
- [x] Brightness census at 0.3 and 0.35. The counts are quoted in the paper sentence below. They are not a thin-cloud class.
- [x] Measure `EMSR260` Campo Galliano and `EMSR286` Ituango South as their own scenes. Do not fold them into 39.3 or 5.6. No clear-cloudy gap is stored for either scene, so they stay out of n=8 and n=32.
- [x] Run `ibm-nasa-geospatial/Prithvi-EO-2.0-300M-TL-Sen1Floods11` on the same protocol. Frozen means the released `WF2_unetv2_all` checkpoint stays unchanged.

## Paper sentences

Add these three sentences. Leave every printed frozen number unchanged.

- [x] Land fill beside the printed 5.6. Keep the printed hard paste at 5.6 [3.0, 8.5], n=32, and the printed test scene mean at 3.8 [1.5, 6.9]. Do not replace printed 5.6 or test 3.8 with 5.4018 or 4.444. Beside them, state the corrected pooled-median check on frozen WF2, seed 42, identical masks, constant fill, scored on unpainted valid pixels. On 32 scenes the cloud fill is `5.401824562730654` `[3.0395970810675137, 8.131108851983626]`, the land fill (non-cloud placebo) is `1.875462012891159` `[0.7947312280281873, 3.0859464397899936]`, and the paired cloud-minus-placebo is `3.5263625498394946` `[1.4732143046767636, 5.72402744697899]`. On 17 test scenes the cloud fill is `4.444186880094727` `[1.6715664311563256, 8.530741356636542]`, the placebo is `2.3272143974326833` `[0.7203564862910251, 4.1754992673787115]`, and the paired contrast is `2.1169724826620437` `[-0.3760086187473995, 5.187757191515272]`. The test paired interval includes 0. Both fills stay constructed fills on clear tiles, scored on unpainted pixels. They are not natural cloud.

- [x] The printed NDVI 1.4 is this UNet. Frozen Prithvi on the same five scenes stays about 26.4, stored mean `26.413202413471293`, interval `[-2.479672216469317, 56.2720287818474]`. Its 32-scene cloud-fill drop is `1.48590782566094` `[0.4696010095725278, 2.557471004375826]`. The placebo drop is larger: non-cloud fill `2.6482709365569335` `[1.8216041771839329, 3.552474710682945]`, paired cloud-minus-placebo `-1.1623631108959933` `[-2.2004243456862635, -0.15842118822902607]`. That smaller cloud-fill drop is not a better-model claim, because the placebo drop is larger.

- [x] Quote the brightness census. Brightness is `sqrt(B2^2+B3^2+B4^2)/10000`. Counts are pixels at or below the cut, then water pixels in that bin. At 0.3, GT-cloud at or below the cut is 1413748 pixels, 237524 water, and above the cut is 710643 pixels, 232857 water; GT-clear at or below the cut is 71202656 pixels, 18076676 water, and above the cut is 4959997 pixels, 2489855 water. At 0.35, GT-cloud at or below the cut is 1951054 pixels, 434810 water, and above the cut is 173337 pixels, 35571 water; GT-clear at or below the cut is 74323427 pixels, 19498563 water, and above the cut is 1839226 pixels, 1067968 water. These counts are not a thin-cloud class.

## Do not merge

Check each line while editing. These quantities stay separate.

- [x] Corrected-rule intervals stay separate from the printed paste. Cloud fill `5.401824562730654` `[3.0395970810675137, 8.131108851983626]`, n=32, is not printed 5.6 [3.0, 8.5]. Test cloud fill `4.444186880094727` `[1.6715664311563256, 8.530741356636542]`, n=17, is not printed test 3.8 [1.5, 6.9].

- [x] The new paste and placebo intervals are scene bootstraps. They are not the event interval [3.1, 8.5] and not the event-weighted 6.0. Those were not recomputed.

- [x] Prithvi and the candidate stay separate from the frozen 39.3 and 1.4. Prithvi unmatched `53.31440092372452` is not 39.3. Prithvi NDVI `26.413202413471293` is not 1.4. Candidate unmatched `34.59240223938893` is not the frozen 39.3. Candidate NDVI `4.294778843697456` is not the frozen 1.4.

- [x] Campo Galliano (`EMSR260_07CAMPOGALLIANO_DEL_MONIT01_v1`, IoU `0.19791666666666666`) and Ituango South (`EMSR286_09ITUANGOSOUTH_DEL_MONIT02_v1`, IoU `0.7203674344866176`) stay out of n=8 and n=32. No clear-cloudy gap is stored for either scene.

- [x] Permanent IoU is not a clear-cloudy gap. Permanent scene-mean IoU `0.9975072173600821` is a stratum score. It is label-dependent. It is not the unmatched 39.3.

## Optional checks

These are still real. They are optional. They are not a reason to train.

- [ ] Event-cluster the new paste and placebo scene gaps from the existing JSON. Do not compute a new mean. The stored intervals are scene bootstraps.

- [ ] Make a qualitative paste figure from real tiles. Caption it as a constructed fill on clear tiles, scored on unpainted pixels. It is not natural cloud. If the tiles are missing, stop.

- [x] Fix the sentence in `results/RUN_REPORT.md` that says "Frozen checkpoints; no additional training,". `results/finetune_v1/` records six epochs. Do not change the released checkpoint.

## Blocked

Permanent-versus-flood stays blocked. `full_genuine_flood_split` is not identifiable from supplied JRC code 0 pixels. Additional independent water-history and flood labels are required. Code 0 is never labelled flood. Do not invent flood labels. The brightness census in item 4 is already done. The missing flood labels are not a reason to train.

## Execution update — 2026-11-15

The three additions are inserted beside the corresponding frozen results in `NOTE.md`; the PDF is not a required input. Markdown and LaTeX insertion text is also available under `results/paper_additions_2026-11-15.*`. The separately referenced `paper/main.tex` is absent, so no edit to that file is claimed. All pre-existing NOTE lines and JSON files were verified unchanged.

The optional event-bootstrap rule and executable script are prepared (`CORRECTED_PASTE_EVENT_RULE.md`, `corrected_paste_event_cluster.py`), but no new interval is claimed until execution passes the stored-scene-interval checks. Colab was disconnected and did not reconnect during this pass. The qualitative figure is not produced: the real rasters are not preserved locally and their availability in the disconnected runtime could not be verified. No substitute imagery, flood labels, or further training was introduced.
