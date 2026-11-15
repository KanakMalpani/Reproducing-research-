# Remaining.md execution status — 2026-11-15

Model improvement is stopped at the user's request. No second improvement candidate was trained. This status concerns the frozen research audit, not the separately authorized training experiment.

| Item | Status | Evidence in results/ |
|---|---|---|
| 1. Colab inputs and environment | Complete | download_manifest.json; runtime_provenance.json |
| 2. Reproduce committed analyses before new results | Complete; three exact JSON matches, 1,291 exact GPU tile confusion matches | reproduction_checks.json; gpu_inference_comparison.json |
| 3. Pooled-median cloud fill and noncloud placebo, seed42 | Complete | pooled_cloud_noncloud_placebo.json; FOLLOWUP_POOLED_PLACEBO_RULE.md in audit folder |
| 4. Permanent/flood split and brightness census | Brightness complete. JRC-supported strata complete. Full genuine-flood comparison blocked by unknown history and absence of independent flood-type labels | brightness_census.json; permanent_nonpermanent_water.json; permanent_flood_completion.json |
| 5. Campo Galliano and Ituango South separately | Complete; kept separate from original gap means | separate_scene_measurements.json |
| 6. Frozen Prithvi on the same protocol | Complete | prithvi_same_protocol.json; prithvi_kill_test_results.json; prithvi_ndvi_comatch_results.json; paired_model_accuracy_comparison.json |
| 7. Frozen original checkpoint, constructed paste | Preserved and verified | original checkpoint SHA256 in finetune_v1/selection.json and permanent_flood_completion.json; no original result JSON replaced |

## Item 4: what can and cannot be concluded

JRC YearlyHistory uses 0=no observations, 1=historically not water, 2=seasonal water and 3=permanent water. The completion script reports each category separately, and reports clear/cloud pixel subsets and coverage. Code1 water is a flood proxy, not independently established flood water. Seasonal water is kept separate. Unknown pixels are never silently labelled flood.

The supplied WorldFloods ground-truth generation code incorporates JRC permanent-water labels. The permanent-water score therefore has label dependence and is not independent evidence of flood-type recognition. Full genuine flood-type evaluation requires independent event-water/history labels for the unknown region. This cannot be completed honestly from the supplied masks alone. No replacement dataset or arbitrary fill was introduced.

Sources: [JRC catalogue](https://developers.google.com/earth-engine/datasets/catalog/JRC_GSW1_4_YearlyHistory), [official GT generation](https://github.com/spaceml-org/ml4floods/blob/192426c7286315b59a2f52c7a9cdebe589242de2/ml4floods/data/create_gt.py), [official JRC acquisition](https://github.com/spaceml-org/ml4floods/blob/192426c7286315b59a2f52c7a9cdebe589242de2/ml4floods/data/ee_download.py).

Unknown GT-water coverage: 9,479,612 pixels, 45.0618% of labelled water.

| JRC category | Eligible scenes | Scene-mean water IoU | 95% scene-bootstrap interval | Secondary pooled IoU |
|---|---:|---:|---|---:|
| unknown | 33 | 0.483510 | [0.40542653338896134, 0.5629346430677318] | 0.673540 |
| flood_proxy_historically_nonwater | 22 | 0.659832 | [0.5636548043777049, 0.7574807625114237] | 0.827485 |
| seasonal | 28 | 0.856121 | [0.7931034029233424, 0.9088180472997666] | 0.909193 |
| permanent | 27 | 0.997507 | [0.994613863041197, 0.9995048042461155] | 0.999230 |

Scene gate: at least200 GT-water pixels; confusion pooled within scenes, unweighted scene means; 5,000 scene-bootstrap draws, seed42. The table describes binary water detection inside JRC spatial strata, not an independently validated multiclass flood model.

## New checklist pass — 2026-11-15

- Added the corrected-rule WF2 placebo comparison beside printed hard-paste values in NOTE.md, keeping printed5.6 and test3.8 unchanged.
- Added the frozen Prithvi comparison beside this UNet's NDVI result; no model-improvement claim.
- Added every requested brightness bin count and the explicit non-thin-class qualification.
- Prepared equivalent Markdown/LaTeX additions in results/paper_additions_2026-11-15.md and .tex; no PDF required. Canonical paper/main.tex is not in this repository; only the available narrative was edited.
- Verified all five separation checks, every original NOTE line, all existing JSON, all new full-precision additions and counts.
- Fixed results/RUN_REPORT.md to distinguish the frozen audit checkpoints from the separately trained six-epoch candidate. Training remains stopped.
- Optional event bootstrap: completed in reconnected Colab using uploaded saved JSON; all source hashes and stored scene intervals verified, means unchanged. See results/EVENT_BOOTSTRAP_FOLLOWUP.md and results/corrected_paste_event_cluster_2026_11_15.json.
- Optional qualitative figure: not produced. The fresh reconnected Colab runtime explicitly reports the old raster and checkpoint paths absent. No raw rasters are stored locally. Stopped at this missing-input boundary; no substitute imagery used.
- Full genuine flood split remains blocked exactly as before.


## Wider data and paper source — completed 2026-11-15

| Requested addition | Status | Evidence |
|---|---|---|
| Actual IEEE-CVF CVPR manuscript source | Complete; newly assembled, not recovered historical source | `../paper/main.tex`, four-page compiled PDF, `../paper/README.md` |
| Preserve every printed frozen quantity and add corrected WF2/activation, Prithvi and brightness sentences | Complete in actual manuscript | `../paper/sec/worldfloods.tex` |
| C2S-MS Sentinel-2 inputs and cloud/water labels | Complete; 900 chips, 8,100 integrity-verified objects, 3,495 eligible windows | `results/c2sms_floods_download_manifest.json`, `results/c2sms_floods_label_preflight.json` |
| Frozen Prithvi composition control | Complete; n=16 event folders for each contrast | `results/c2sms_floods_prithvi_composition.json` |
| C2S-MS constructed cloud versus land fill | Complete; n=18 event folders, paired interval includes zero | `results/c2sms_floods_prithvi_same_tile.json` |
| Explicit unweighted event-folder scene aggregator and intervals | Complete | `results/c2sms_floods_prithvi_aggregator.json` |
| Preserve old JSON and released weights | Verified; all 65 pre-existing tracked JSON unchanged; checkpoint SHA256 unchanged | `results/C2SMS_VERIFICATION.md`, runtime/execution records |

The C2S-MS NDVI-matched gap remains positive, unlike the frozen WorldFloods UNet result; no cross-dataset near-zero-gap claim is made. No training was performed. See `results/C2SMS_FROZEN_RUN_REPORT.md` for full-precision results, event exclusions and limits. Only the optional missing-raster WorldFloods figure and independently labelled genuine-flood split remain unresolved.
