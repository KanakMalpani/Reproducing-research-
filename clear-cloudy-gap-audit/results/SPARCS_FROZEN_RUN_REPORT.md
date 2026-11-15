# SPARCS: completed frozen third-dataset stress test

Dataset: **Landsat 8 SPARCS**, DOI **10.5066/F7FB5146**. Executed in the user's Google Colab T4 runtime. No training, fine-tuning, new segmenter, sensor harmonization or threshold selection. No WorldFloods or C2S-MS result JSON was replaced.

This test adds manual water/cloud-labelled Landsat data. It is a cross-sensor, visible-surface stress test, not an independent flood-event benchmark or an exact Sentinel-2 replication. The same released Prithvi checkpoint is held fixed.

## Inputs and fixed adaptations

All 80 original subscenes and 80 original acquisitions have imagery, manual mask and acquisition metadata. The official USGS archive contains 1555733976 bytes, observed SHA256 `5cde604615ee241950b5a0b641ae5de82b27738b51e39aa724d18faec0bbaa9a`. All ZIP CRCs passed. This observed SHA256 is not presented as a publisher-certified checksum. Every extracted member is recorded with size/SHA256.

The first downloader stopped because metadata filenames omit the numeric subscene suffix. The original metadata was present. Commit `9a21da9` fixed that mapping before the census, explicitly resumed the existing archive and rechecked CRCs; no missing label/metadata was fabricated. The analysis protocol was published in `b43794d` and evaluator in `a1d44e2`, before predictions. The filename correction changes neither the sample nor statistical rules.

Landsat B2/B3/B4/B5/B6/B7 are mapped to blue/green/red/NIR/SWIR1/SWIR2. Each scene's MTL coefficients and solar elevation convert original DN to TOA reflectance; no clipping or fitting is applied. The official model normalization stays unchanged. Imagery remains at native30m: a 256-pixel input has a different physical extent from Sentinel-2. Bottom/right partial windows are zero-padded only outside the image and padded pixels never count in denominators or scoring.

Manual cloud5 is excluded from water scoring, because its underlying water status is unknown. Water2 and flooded6 are positive; land4 negative. Shadow0/1 and snow3 are excluded. The cloud-fraction denominator uses image-valid2/4/5/6 independently of visible water scoring. Thus this natural contrast is visible-water IoU on clear versus cloudy windows; it is not accuracy beneath cloud and not the all-label C2S-MS estimand.

The fixed nonoverlapping window census has 1280 candidates and 910 eligible windows. Exclusion reasons (may overlap): `{'less_than_half_visible_water_label_valid': 370, 'less_than_half_image_and_cloud_label_valid': 81}`. Complete per-scene calibration, geometry and class census are in `sparcs_label_preflight.json`.

Sources: [USGS manual labels/data](https://www.usgs.gov/landsat-missions/spatial-procedures-automated-removal-cloud-and-shadow-sparcs-validation-data), [original laboratory description](https://emapr.ceoas.oregonstate.edu/sparcs/), [USGS DN calibration](https://www.usgs.gov/landsat-missions/using-usgs-landsat-level-1-data-product).

## Measured controls and aggregator

Primary aggregator: unweighted mean across eligible original acquisitions, pooling window confusions within an acquisition first. Intervals:5,000 acquisition bootstrap draws, seed0 composition and42 fills; no new event-weighted point is inferred without event IDs. Zero/one-unit limitations and all exclusions are preserved. Points and intervals below are copied at stored precision.

| Comparison | Eligible acquisitions | Mean points | 95% interval |
|---|---:|---:|---|
| Unmatched visible-water gap | 25 | 10.327575483130197 | [-1.646243507476738, 22.72633927445182] |
| Water-fraction matched | 9 | -8.343762771367127 | [-24.072823614667662, 5.2871293459313335] |
| Water fraction + NDVI | 8 | -6.79222020686342 | [-21.025566522896437, 2.6992338951401873] |
| Constructed cloud-fill drop | 45 | 2.2375298353549113 | [0.790032331068722, 4.164813926239308] |
| Constructed land-fill drop | 45 | 1.3865834373515544 | [0.6806836616869414, 2.2788597823701027] |
| Paired cloud minus land | 45 | 0.8509463980033569 | [-0.2382887005613511, 2.4214038171739944] |

Matching uses the same greedy water/NDVI tolerances (.05 each) and minimum200 GT-water-pixel group support as the fixed control implementation. The same-tile comparison derives both spectra and donor masks only from SPARCS, with identical masks and a common visible labelled unpainted surface. Both fills are constructed constant fills on clear windows, scored on unpainted pixels; they are not natural cloud.

Secondary pixel-pooled composition gaps (not primary scene means): {'unmatched': -3.1769746014943867, 'water_fraction_match': -0.6828959133840962, 'water_fraction_plus_ndvi_match': -0.2539720745156848}.

Descriptive visible-water accuracy, separate from gaps: {'scene_count': 62, 'scene_mean': 0.5185537355563729, 'ci95': [0.43726768253741155, 0.598654200380362], 'uncertainty': 'acquisition scene bootstrap; 5000 draws; seed0', 'limitation': None, 'pixel_pooled_water_iou_secondary': 0.7210599831494662}.

## Interpretation and limits

All three composition intervals include zero. The NDVI-matched point is -6.79222020686342 [-21.025566522896437, 2.6992338951401873], n=8 original acquisitions. This does not establish either a positive gap or equivalence to zero. Retention changes from 25 unmatched acquisitions to 9 water-matched and 8 NDVI-matched; subtracting their points is not a controlled effect estimate on one fixed scene population.

The paired cloud-minus-land fill contrast is 0.8509463980033569 [-0.2382887005613511, 2.4214038171739944], n=45. Its interval also includes zero. Both individual fills have positive mean drops, but this run does not establish an additional positive cloud-spectrum penalty over the land placebo.

This differs from the positive C2S-MS matched Prithvi result. It does not overturn either earlier result: the sensor, spatial support and water-label estimand differ. The findings do not justify a universal numerical gap across datasets.

The table reports what the third dataset supports; no result was adjusted to agree with a previous dataset. A remaining observational gap cannot be assigned to cloud causality: composition matching does not remove every landcover/scene difference. Paired fills measure sensitivity to constant replacement, not physical clouds.

Sensor spectral responses,30m resolution, calibration, excluded snow/shadows and missing cloud-covered water labels differ from Sentinel-2. Consequently this run cannot isolate a sensor effect from a dataset/label effect, or establish general flood accuracy. Manual masks were subjectively digitized by one analyst, as noted by USGS. Checkpoint pretraining overlap is unverified. A small fill drop alone is not model superiority.

## Preservation and verification

Frozen checkpoint revision `91ce9d38086a80b078a192b374df758b8855b732`; SHA256 before/after `76eed77d8bd543ae441308b80e8408502243a871cb8a338ddc8220ed96dfc270`. Strict loading, FP32, eval/no gradients, batch2, no TTA/location/time metadata. Runtime/environment and executed code hashes are recorded. The shared fixed matching/bootstrap helper is verified by SHA256 before import; no C2S-MS raster/label/spectrum/prediction is used.

`sparcs_verify_results.py` checks source/input hashes, pair uniqueness/tolerances, eligible counts, confusion conservation, identical common scoring-surface water totals and exact copied summaries. It verifies every previously tracked JSON against the fixed pre-results baseline. Local checks perform no inference or bootstrap.

The original archive TIFFs trigger GDAL photometric/ExtraSamples metadata warnings for their multiband layout. All selected bands are read by explicit index; geometry/calibration/labels and finite logits are checked. Warnings remain in the execution log; input imagery was not rewritten.

Original WorldFloods/C2S-MS values remain unchanged. The optional missing-raster WorldFloods illustration and genuine flood split remain unresolved. This stress test does not substitute for either.
