# C2S-MS Floods: completed frozen three-control run

Executed in the user's reconnected Google Colab T4 runtime on 2026-11-15. No training, fine-tuning, additional segmenter, threshold optimization or modification of WF2 occurred. Dataset: **C2S-MS Floods**, DOI **10.34911/rdnt.oz32gz**.

The protocol was committed before the label census or predictions (`4b15a07ae5ad1491aa78a4dbca9bd2213a13bacf`); the evaluated inference code is pinned at `e79db6acacae62c405ccd3daf2d1d12ea343bc41`. See `../C2SMS_FROZEN_CONTROL_RULE.md` for every fixed threshold, donor rule, scoring mask, spectrum and aggregation choice.

## Actual inputs and coverage

All 900 Sentinel-2 chip folders from all 18 event folders had the seven required bands plus LabelCloud and LabelWater: 8,100 objects, each checked against object size and simple-MD5 ETag, with SHA256 recorded. The 62 source acquisitions are recorded separately; an event folder is the scoring/bootstrapping unit, not an independent acquisition.

All seven stored band arrays were 512 x 512 on aligned extents/CRS; labels were checked for exact alignment and legal codes. The predeclared reader supports nearest-neighbour band alignment, but no smaller stored array required upsampling in this inventory. Four nonoverlapping 256 x 256 windows per chip gave 3,600 candidates; 3,495 passed the two 50% validity gates, and 105 failed both. Eligible windows: 1,802 clear, 1,195 cloudy, 498 intermediate (excluded from clear/cloudy contrasts).

LabelCloud 0/1 define clear/cloud; 2 (shadow) and 255 (no data) are excluded from scoring. LabelWater 0/1 supply land/water; 255 is excluded. Cloud fraction is measured independently of water-label availability. The census has 182,937,704 image-valid clear pixels and 38,200,265 image-valid cloud pixels with known water labels; neither group contains image-valid water-label no-data pixels. The 8,206,916 image-valid shadow pixels remain excluded despite having water labels. These are overlapping-chip inventory counts, not deduplicated geographic area.

Dataset sources: [official AWS registry](https://registry.opendata.aws/c2smsfloods/) and [publisher documentation](https://radiantearth.blob.core.windows.net/mlhub/c2smsfloods/Documentation.pdf).

## Measured controls

Primary aggregator: **unweighted mean of eligible event-folder scene values**. Confusions are pooled within an event before IoU/gap calculation. Intervals are 5,000 event-folder bootstrap draws, seed 0 for composition and seed 42 for fills. The table below copies the stored full-precision point estimates and intervals.

| Comparison | Eligible event folders | Mean gap/drop, points | 95% interval |
|---|---:|---:|---|
| Unmatched clear minus cloudy | 16 | 27.758077975674098 | [15.334126064499115, 39.016190238008484] |
| Water-fraction matched | 16 | 21.141924169417972 | [10.89209165835164, 32.240721035769106] |
| Water fraction + NDVI matched | 16 | 25.40583933039207 | [13.919263631579188, 36.133004262915804] |
| Constructed cloud fill | 18 | 0.5202762097675461 | [0.21480288916273826, 0.8747717819543489] |
| Constructed non-cloud land fill | 18 | 0.7116918145604925 | [0.059713091840471764, 1.6704480408301774] |
| Paired cloud minus land fill | 18 | -0.19141560479294636 | [-1.0693382967417622, 0.35248639400358794] |

All composition modes exclude the same two events: `55d43c77-9955-4b4f-b6f0-b21e7f50a039` and `e7e6e693-aec9-44b2-928d-80ba818b647e`, which lack an eligible cloudy group. Water matching retains 538 unique pairs; NDVI co-matching retains 422. Exact pairs, tolerance differences, missingness and dropped groups are in the composition JSON.

The same-tile comparison uses all 1,802 clear windows and retains all 18 events. Both spectra are exact C2S-MS six-band medians under the locked rule; non-cloud land explicitly requires LabelCloud 0 and LabelWater 0. The cloud and land fills have identical donor masks and an identical common unpainted LabelCloud-0 scoring surface. There are 369 target windows with a borrowed C2S-MS event mask. Every event has identical scored GT-water totals for original/cloud/land predictions. Caption: **Constructed constant fill on clear C2S-MS tiles, scored on unpainted pixels; it is not natural cloud.**

Secondary pixel-pooled composition gaps are 35.98125085696435, 23.57837469034012, 28.18432778375118 points respectively. They are not the scene means. A separate event-weighted point is not estimated because the primary scene unit already is an event folder.

Descriptive overall water-IoU scene mean: 0.6018354474856003 [0.5074719414119634, 0.688569051715337], n=18 events. Secondary pixel-pool IoU: 0.6436905374529545. These are accuracy scores, not gaps and not evidence of a model improvement.

## Interpretation

The three controls apply on the wider dataset, but the numerical WorldFloods UNet conclusion does not carry over: C2S-MS Prithvi retains a positive matched gap, including after NDVI co-matching. Matching did not remove that gap under this fixed protocol. This is a different dataset and model, so it does not isolate which change explains the difference.

The paired cloud-minus-land fill interval includes zero. A cloud-specific penalty beyond constant land replacement is not established. The small cloud-fill drop alone does not establish robustness or superiority. The natural unmatched and matched gaps remain observational and composition/date/location confounded.

Within-event matching is broader than within-acquisition matching. Chips may overlap geographically and are not deduplicated; event-level bootstrap retains within-event dependence but 18 candidate events give limited uncertainty resolution. Training-event overlap with Sen1Floods11 has not been ruled out, so this is descriptive transfer evaluation, not a verified independent blind benchmark. All valid clear/cloud water labels are accepted as released; physical visibility/label uncertainty beneath cloud is not independently validated.

WorldFloods uses a GT-clear non-cloud median that can include water; this C2S-MS placebo is land-only. The fill comparisons share a control structure but are not identical operational interventions. No natural-cloud causality, thin-cloud class, flood-type recognition or model-improvement claim follows.

## Reproduction and verification

- Frozen model: `ibm-nasa-geospatial/Prithvi-EO-2.0-300M-TL-Sen1Floods11`, revision `91ce9d38086a80b078a192b374df758b8855b732`.
- Checkpoint SHA256 before and after inference: `76eed77d8bd543ae441308b80e8408502243a871cb8a338ddc8220ed96dfc270`.
- Isolated Python 3.11.18, TerraTorch 0.99.8, torch 2.6.0+cu124, FP32, batch 2, strict full-state load, evaluation/no-gradient mode, two-class argmax, official Sen1Floods11 normalization, no TTA or location/time metadata.
- Inference/control runtime: 681.4900999069214 seconds (excludes download/environment/preflight). Environment freeze and GPU/runtime provenance are retained.
- Downloaded ZIP SHA256: `6966eb5897ff20a5cd7bbe9d39d4239f532f7873413a256c788ee0b352d4dcf4`; size 2,628,975 bytes. Exported predictions are bit-packed in `c2sms_floods_original_predictions.npz`.
- Verified all fixed matching tolerances, pair uniqueness, eligible counts, tile confusion accounting, identical common-surface water totals, exact copied summaries and cloud/local script SHA256. All 65 previously tracked JSON files remain unchanged (Git line endings normalized for comparison).
- The preflight inventory SHA256 names the exact LF-encoded cloud copy. The preserved Windows working copy differs only by CRLF; JSON content and normalized bytes match. The original working file was not overwritten.

New required files are `c2sms_floods_prithvi_composition.json`, `c2sms_floods_prithvi_same_tile.json`, and `c2sms_floods_prithvi_aggregator.json`. Supporting files retain the input/download inventory, preflight/census, per-window confusion, original predictions, environment, execution log and runtime provenance. All JSON names the dataset and DOI.

The actual CVPR-format paper is newly assembled at `../../paper/main.tex`; it includes the original frozen values, corrected WF2 placebo/activation intervals, Prithvi, brightness census and these wider-data results. It is not a recovered historical manuscript. Optional WorldFloods real-tile illustration remains stopped at missing rasters; genuine flood split remains blocked by missing independent history/flood labels. No substitutions were made.
