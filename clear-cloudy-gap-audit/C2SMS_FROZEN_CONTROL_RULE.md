# C2S-MS Floods: frozen Prithvi three-control protocol

Locked before any C2S-MS model predictions or label census. Dataset: **C2S-MS Floods**, DOI **10.34911/rdnt.oz32gz**. No training, fine-tuning, threshold selection, new segmenter, invented labels or modification of WorldFloods results is permitted. This is descriptive transfer evaluation, not a new blind benchmark; overlap between the checkpoint's training events and this dataset has not been established.

## Inputs and geometry

Use the complete public `radiant-mlhub/c2smsfloods/chips/` Sentinel-2 inventory. The listing observed 900 chip folders in18 event folders, each containing all9 required files: B2, B3, B4, B8, B8A, B11, B12, LabelWater and LabelCloud GeoTIFFs. This inventory count is not an eligible scene count. Preserve object keys, sizes and ETags; verify sizes/ETag MD5 and record SHA256 on download. Stop on a missing required chip, label, band or failed integrity check. No Sentinel-1 or WorldFloods imagery/label may be substituted.

Use B2 as the512x512 target grid. Other reflectance bands must have the same CRS/extent and are resampled to that grid with nearest-neighbour interpolation, preserving integer DN. This explicitly handles native20m bands. Labels must already align exactly with B2; otherwise stop. Zero DN is image no data. Keep the publisher's DN scaling; divide by10000 for model reflectance, with no clipping or radiometric tuning. Valid model imagery requires all six input bands to be nonzero. Native chips are split into four nonoverlapping256px windows; inference is at256px without resizing/TTA. Dataset chip overlaps, if any, are not deduplicated.

## Model

Use only `ibm-nasa-geospatial/Prithvi-EO-2.0-300M-TL-Sen1Floods11`, revision `91ce9d38086a80b078a192b374df758b8855b732`. Checkpoint SHA256 `76eed77d8bd543ae441308b80e8408502243a871cb8a338ddc8220ed96dfc270`. Load the complete state strictly with TerraTorch0.99.8, `backbone_pretrained=False`, evaluation mode, no gradients, no temporal/location metadata as configured. Inputs: B2,B3,B4,B8A,B11,B12, official Sen1Floods11 means/stds, two-class argmax. Verify the checkpoint hash remains unchanged after inference. WF2 is not loaded or modified.

## Labels, eligibility and grouping

LabelCloud:0=clear/background,1=cloud,2=shadow,255=no data. LabelWater:0=not water,1=water,255=no data. Reject unexpected codes. Exclude cloud shadow and either label's no data from scores. Count label availability by cloud code; do not invent water labels beneath clouds.

A tile must have at least50% pixels with valid imagery and LabelCloud in{0,1}, and at least50% jointly scoreable water/cloud labels. Cloud fraction uses the first mask, independently of water-label availability; it must not silently become zero when cloud pixels lack water labels. Clear tiles have cloud fraction<.01, cloudy tiles>=.1; intermediate tiles are reported but outside the contrast. Water fraction uses jointly scoreable pixels. NDVI is the mean of (B8-B4)/(B8+B4+1e-6) on scoreable LabelCloud0 pixels with nonzero B8; no such pixels means missing NDVI.

The primary scoring unit called a scene in output is an **event folder**, pooling its chips/windows. This deliberate event-level grouping is not a count of independent acquisitions. Retain source acquisition IDs in tile metadata and report their count separately. Within-event matching is broader than within-acquisition matching and cannot remove all date/location differences. No WorldFloods sample count is imposed.

## Composition control

Within each event folder, pool pixel confusions over clear tiles and cloudy tiles separately; require>=200 GT-water pixels in each pooled group and defined IoUs. Unmatched gap is100*(clear IoU-cloudy IoU). Replay the original deterministic greedy matching: cloudy tiles ascending water fraction, unused clear candidate with smallest absolute water-fraction difference, tolerance.05; tie by fixed metadata order. For NDVI co-match require both absolute differences<=.05 and choose minimum sum of differences, then fixed order. Skip/count missing NDVI. Pool matched-group confusions within event, retaining the200 GT-water gate. Record tile/pair counts and all exclusion reasons, not just surviving events.

## Same-tile control

Both spectra come only from C2S-MS. Per-event cloud fill: exact per-band median across image-valid LabelCloud1 pixels. Noncloud **land** fill: exact median across image-valid LabelCloud0 and LabelWater0 pixels. Use six-band integer histograms after nearest-neighbour resampling. Require>=1000 eligible pixels; otherwise use the frozen pooled C2S-MS median of that class, never a WorldFloods spectrum. Record sources and pixel counts.

For each event's clear tiles in fixed order, use its own donor masks with cloud fraction in[.1,.6], cyclically; if absent use the lexicographically ordered full-dataset donor pool at index(i+42)%N. Masks are LabelCloud1 only. The cloud and land fills use the identical mask. Evaluate original and both painted predictions only on image-valid, LabelWater0/1, LabelCloud0, unpainted pixels. Require>=200 GT-water pixels on this common scoring surface, and defined IoUs, per event. Record dropped events. No training or inference is scored on painted pixels. The paste is a **constructed constant fill on clear tiles, scored on unpainted pixels; it is not natural cloud**.

## Aggregation and files

For each contrast, the primary point is the unweighted mean of eligible event-folder scene gaps. Interval:5000 ordinary scene/event-folder bootstrap draws, seed0 for composition and42 for paste, percentile2.5/97.5, fresh RNG per statistic. With only one eligible event, label the degenerate interval descriptive; with zero, emit null with a reason. Event folders are the bootstrap units. Do not calculate a separate event-weighted point because it would duplicate the primary unit; pixel-pooled contrasts are secondary and explicitly named. Report event counts from the data and all grouping/gating choices.

Write new JSON only under results/, using exclusive creation: c2sms_floods_prithvi_composition.json, c2sms_floods_prithvi_same_tile.json, c2sms_floods_prithvi_aggregator.json, plus named input/download/tile/census provenance. Every C2S-MS result names the dataset and DOI. Never insert these into any WorldFloods JSON or n=8,n=7,n=5,n=32. Inspect input geometry/label availability before model execution; if the controls cannot be estimated, report that fact rather than changing this rule.

Primary sources: https://registry.opendata.aws/c2smsfloods/ ; https://radiantearth.blob.core.windows.net/mlhub/c2smsfloods/Documentation.pdf ; pinned Prithvi model card/configuration and the existing audited Sen1Floods11 loading code.
