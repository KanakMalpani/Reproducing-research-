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

- [x] Event-cluster the new paste and placebo scene gaps from the existing JSON. Do not compute a new mean. The stored intervals are scene bootstraps.

- [ ] Make a qualitative paste figure from real tiles. Caption it as a constructed fill on clear tiles, scored on unpainted pixels. It is not natural cloud. If the tiles are missing, stop.

- [x] Fix the sentence in `results/RUN_REPORT.md` that says "Frozen checkpoints; no additional training,". `results/finetune_v1/` records six epochs. Do not change the released checkpoint.

## Blocked

Permanent-versus-flood stays blocked. `full_genuine_flood_split` is not identifiable from supplied JRC code 0 pixels. Additional independent water-history and flood labels are required. Code 0 is never labelled flood. Do not invent flood labels. The brightness census in item 4 is already done. The missing flood labels are not a reason to train.

## Execution update — 2026-11-15

The three additions are inserted beside the corresponding frozen results in `NOTE.md`; the PDF is not a required input. Markdown and LaTeX insertion text is also available under `results/paper_additions_2026-11-15.*`. The separately referenced `paper/main.tex` is absent, so no edit to that file is claimed. All pre-existing NOTE lines and JSON files were verified unchanged.

The optional event bootstrap has now executed in the reconnected Colab session using the saved JSON inputs uploaded into a separate follow-up folder. All stored scene intervals and source hashes passed verification, and point estimates were copied unchanged. See `results/EVENT_BOOTSTRAP_FOLLOWUP.md` and `results/corrected_paste_event_cluster_2026_11_15.json` for the separately labelled activation-clustered intervals. The original event interval and event-weighted point were not recomputed.

The qualitative figure remains stopped at the missing-input boundary: the reconnected runtime is fresh, and a file preflight found the original audit folder, /content/worldfloods rasters and /content/wf-models/model.pt absent. Local storage contains the weights and manifests but not the raw satellite rasters. No substitute imagery, invented flood labels, or further training was introduced.

## What the paper should contain

Checklist for the IEEE-CVF CVPR source. It is not a list of new experiments. `results/paper_additions_2026-11-15.md` and `results/paper_additions_2026-11-15.tex` in this folder are drafts. They are not the paper. `paper/main.tex` was absent when this checklist was written. Newly assembled IEEE-CVF CVPR source now exists at `../paper/main.tex`, including the activation intervals below; the insertion drafts remain historical supporting material. The two checked lines are constraints, not open experiments.

- [x] Keep every printed frozen number. Name the composition control, the same-tile control, and the aggregator. Printed frozen quantities: unmatched 39.3 [22.4, 57.6] n=8; water-fraction 8.9 [-0.7, 17.8] n=7; NDVI 1.4 [-24.5, 24.5] n=5; hard paste 5.6 [3.0, 8.5] n=32, median 2.3; test 3.8 [1.5, 6.9]; pixel-pool 2.8; event interval [3.1, 8.5]; event-weighted mean 6.0; corrections −0.24 and +0.44. The composition control is the water-fraction match 8.9 [-0.7, 17.8] n=7 together with the NDVI co-match 1.4 [-24.5, 24.5] n=5; unmatched 39.3 [22.4, 57.6] n=8 stays beside it. The same-tile control is the hard paste 5.6 [3.0, 8.5] n=32, median 2.3, and the test scene mean 3.8 [1.5, 6.9]. The aggregator is the unweighted scene mean of that paste, named apart from the pixel-pool 2.8, the event interval [3.1, 8.5], and the event-weighted mean 6.0.

- [x] Add the land fill beside printed 5.6, and add the activation interval on the paired difference. Do not replace 5.6, 3.8, [3.1, 8.5], or 6.0. On frozen WF2, 32 scenes, the corrected-rule cloud fill is `5.401824562730654` `[3.0395970810675137, 8.131108851983626]` and the land fill is `1.875462012891159` `[0.7947312280281873, 3.0859464397899936]`. The paired difference has 32 scenes and 24 activations. Its stored mean `3.5263625498394946` is unchanged, and its scene interval stays `[1.4732143046767636, 5.72402744697899]`. The activation interval in `results/corrected_paste_event_cluster_2026_11_15.json` is `[1.4415350645592608, 5.984003104125222]`. That interval is not printed 5.6 [3.0, 8.5] and not [3.1, 8.5], and its upper end `5.984003104125222` is not the event-weighted 6.0. The cloud-fill activation interval `[3.075068940840618, 8.185912766151164]` is not [3.1, 8.5]; the stored cloud mean remains `5.401824562730654`. On test, 17 scenes and 11 activations, the paired activation interval `[-0.38313817636896463, 5.5367285405207305]` still includes 0. The test cloud stored mean stays `4.444186880094727`, which is not printed test 3.8. Both fills are constructed fills on unpainted pixels, not natural cloud.

- [x] State that printed NDVI 1.4 [-24.5, 24.5] n=5 is this UNet. Frozen Prithvi on the same five scenes has mean `26.413202413471293` and interval `[-2.479672216469317, 56.2720287818474]`. Its 32-scene cloud-fill drop is `1.48590782566094` `[0.4696010095725278, 2.557471004375826]`. The non-cloud fill drop is larger: `2.6482709365569335` `[1.8216041771839329, 3.552474710682945]`, paired `-1.1623631108959933` `[-2.2004243456862635, -0.15842118822902607]`. The smaller cloud-fill drop is not a better-model claim.

- [x] Quote the 0.3 and 0.35 brightness counts as a census, not a thin-cloud class. Brightness is `sqrt(B2^2+B3^2+B4^2)/10000`. Counts are pixels at or below the cut, then water pixels in that bin. At 0.3, GT-cloud at or below the cut is 1413748 pixels, 237524 water, and above the cut is 710643 pixels, 232857 water; GT-clear at or below the cut is 71202656 pixels, 18076676 water, and above the cut is 4959997 pixels, 2489855 water. At 0.35, GT-cloud at or below the cut is 1951054 pixels, 434810 water, and above the cut is 173337 pixels, 35571 water; GT-clear at or below the cut is 74323427 pixels, 19498563 water, and above the cut is 1839226 pixels, 1067968 water.

- [ ] One qualitative figure from real tiles. Caption it as a constructed fill on clear tiles, scored on unpainted pixels. It is not natural cloud. Rasters are absent, so this box stays open.

- [x] Do not train. The six-epoch candidate missed its 0.9 test-IoU target. Test cloudy IoU fell from `0.7667548616804217` to `0.7390338550118456`. That result is not a paper improvement.

- [x] The flood split stays out until independent flood labels exist. `full_genuine_flood_split` is not identifiable from supplied JRC code 0 pixels.

## Wider data

This is the only open experiment that changes the claim. The paper is what is being built. This run tests that the composition control, the same-tile control, and the aggregator still apply on another dataset. A checkpoint is scored only under those three controls. Do not train. Do not add a model improvement or a new segmenter. Do not train a WorldFloods model. Do not replace a printed WorldFloods number. Do not predict an IoU, a scene count, or an interval. Write new JSON only under `results/`. Each file names `C2S-MS Floods` and stays out of WorldFloods n=8, n=7, n=5, and n=32.

A real-tile constructed-fill figure on WorldFloods stays optional and does not replace this run. Flood labels stay blocked.

- [x] Get the C2S-MS Floods Sentinel-2 chips and the dataset cloud label `LabelCloud`. Registry: https://registry.opendata.aws/c2smsfloods/ . DOI: `10.34911/rdnt.oz32gz`. List the bucket with `aws s3 ls --no-sign-request s3://radiant-mlhub/c2smsfloods/`. `LabelCloud` is the Sentinel-2 cloud GeoTIFF: 0 background, 1 cloud, 2 cloud shadow, 255 no data. Water is `LabelWater`: 0 not water, 1 water, 255 no data. Sentinel-1 chips have no `LabelCloud`. If the Sentinel-2 tiles or `LabelCloud` are missing, stop. Do not substitute WorldFloods tiles or a WorldFloods cloud fraction.

- [x] Score `ibm-nasa-geospatial/Prithvi-EO-2.0-300M-TL-Sen1Floods11` frozen on those chips. Do not train it. Do not fine-tune it on C2S-MS Floods or on WorldFloods. Do not add a segmenter. Leave `WF2_unetv2_all` unchanged. The checkpoint is only an input to the three controls below.

- [x] Composition control. On C2S-MS Floods, using `LabelCloud` and `LabelWater`, compute unmatched clear-versus-cloudy water IoU, then the water-fraction match, then water-fraction plus NDVI. Clear and cloudy come from `LabelCloud` codes 0 and 1. Leave cloud shadow (code 2) and no data (255) out of both. Write `results/c2sms_floods_prithvi_composition.json` with `"dataset": "C2S-MS Floods"` and `"doi": "10.34911/rdnt.oz32gz"`. Leave IoU, scene count, and interval unset until this run writes them. Do not copy WorldFloods 39.3, 8.9, 1.4, n=8, n=7, or n=5.

- [x] Same-tile control. On clear tiles, apply a constructed constant fill. Compare a cloud spectrum with a non-cloud land fill. Take both fills from C2S-MS chips. Score unpainted pixels. The fill is a constructed fill on clear tiles, scored on unpainted pixels. It is not natural cloud. Write `results/c2sms_floods_prithvi_same_tile.json` with `"dataset": "C2S-MS Floods"`. Leave the gaps unset until this run writes them. Do not copy WorldFloods 5.6, 3.8, or n=32. If the tiles or `LabelCloud` are missing, stop.

- [x] Aggregator. Name the aggregator in `results/c2sms_floods_prithvi_aggregator.json`: the scene mean and the interval. On WorldFloods that aggregator is the unweighted scene mean, named apart from the pixel-pool and the event-weighted mean. Compute that aggregator on C2S-MS event folders. Set `"dataset": "C2S-MS Floods"`. Write the scene count the chips support. Do not set it to WorldFloods n=8, n=7, n=5, or n=32. Do not reuse a printed WorldFloods interval.


## Wider-data and actual paper completion — 2026-11-15

The only new experiment is complete in the reconnected Colab T4 session: all 900 C2S-MS Sentinel-2 chips (8,100 required objects) verified, 3,495 eligible windows, 18 event folders, 62 source acquisitions. Frozen Prithvi was scored under the locked composition, same-tile and aggregation controls. All three required result JSON files now contain measured estimates, actual eligible counts and intervals, plus the dataset and DOI. See `results/C2SMS_FROZEN_RUN_REPORT.md` and the new input/preflight/tile/environment/runtime records.

Composition means are 27.758077975674098 unmatched, 21.141924169417972 water-matched and 25.40583933039207 with NDVI, each n=16 event folders. The paired cloud-minus-land fill mean is -0.19141560479294636 with interval [-1.0693382967417622, 0.35248639400358794], n=18 event folders. Matching does not remove the C2S-MS Prithvi gap, and the paired fill interval includes zero. These results are separate from WorldFloods and are not model-improvement claims.

Actual source is now newly assembled at `../paper/main.tex` using the unchanged official CVPR author-kit style. It preserves the printed frozen numbers and contains all four requested numerical additions, the activation intervals, and measured wider-data results. It compiled to a four-page PDF and was visually checked. This is newly built paper source, not a recovered historical manuscript or a claimed conference submission. No PDF was required as an input. All 65 previously tracked JSON files were verified unchanged, and the frozen Prithvi checkpoint hash is unchanged.

The optional WorldFloods qualitative figure remains open due to absent rasters. The genuine flood split remains blocked by missing independent labels. No training, substitute imagery or invented flood labels were introduced.


## Additional requested dataset — SPARCS completed 2026-11-15

At the user's request, an additional frozen Prithvi test ran in Colab on Landsat 8 SPARCS (DOI10.5066/F7FB5146), all80 subscenes/acquisitions,910 eligible windows. This is a separately labelled cross-sensor, visible-water stress test; it does not replace the Sentinel-2 flood experiments.

All three controls completed. Unmatched10.327575483130197 [-1.646243507476738,22.72633927445182], n=25; water-matched-8.343762771367127 [-24.072823614667662,5.2871293459313335], n=9; NDVI-matched-6.79222020686342 [-21.025566522896437,2.6992338951401873], n=8. Paired cloud-minus-land fill0.8509463980033569 [-0.2382887005613511,2.4214038171739944], n=45. All these intervals include zero; differing retained samples prevent interpreting the point changes as an adjustment effect on one fixed population. No equivalence or universal-effect claim is made.

See `results/SPARCS_FROZEN_RUN_REPORT.md`, `results/SPARCS_VERIFICATION.md`, the new `sparcs_*.json`, `SPARCS_RUN.md` and `../paper/sec/sparcs.tex`. All72 previously tracked JSON files and earlier numerical manuscript sections remain unchanged. Updated actual paper/PDF compiled and was visually checked. No training. Missing-input WorldFloods figure and genuine flood split boundaries are unchanged.
