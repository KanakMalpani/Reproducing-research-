# WorldFloods controlled audit: execution report

## Completed work

- Downloaded 102 validation/test rasters (about 5.24 GiB), WF2 weights and Prithvi weights; checksums and revisions are saved.
- Recomputed the three required analyses: exact JSON matches.
- Reran WF2 inference on all 1,291 tiles and 34 scenes: every confusion count matched.
- Ran corrected pooled-median and non-cloud paste controls, JRC water strata, brightness census, two separate scene checks and frozen Prithvi evaluation.

## Constructed paste controls

IoU drops in percentage points. Unweighted scene mean; 5,000 scene-bootstrap draws, seed 42; 95% intervals. Same masks and scoring surface for both fills.

| Model | Cloud fill | Non-cloud fill | Paired cloud minus placebo |
|---|---:|---:|---:|
| WF2_UNet | 5.402 [3.04, 8.131] | 1.875 [0.795, 3.086] | 3.526 [1.473, 5.724] |
| Prithvi_Sen1Floods11 | 1.486 [0.47, 2.557] | 2.648 [1.822, 3.552] | -1.162 [-2.2, -0.158] |

The corrected WF2 run uses exact pooled cloud-pixel medians and the written fallback donor fraction filter. The committed implementation sampled pixels, took a median of scene medians for the frozen fallback, and disabled the fallback donor filter with an always-true condition. This is a follow-up correction; it does not replace the committed 5.59-point result. The placebo rule was fixed before computing the result.

The paste is a constructed constant fill on clear tiles, scored on unpainted labelled pixels. These numbers do not establish natural-cloud causality.

## Natural composition controls

| Model | Unmatched gap | NDVI co-match gap |
|---|---:|---:|
| WF2_UNet | 39.335 [22.442, 57.577] (n=8) | 1.427 [-24.531, 24.515] (n=5) |
| Prithvi_Sen1Floods11 | 53.314 [34.997, 73.337] (n=8) | 26.413 [-2.48, 56.272] (n=5) |

Both models use unchanged terrain metadata, within-scene matching and named scene aggregation. Pixel-pooled comparisons are secondary and saved separately. A smaller paste drop alone does not establish a better model.

## JRC strata, brightness and separate scenes

JRC==3 is permanent; codes 1/2 are known nonpermanent; code 0 is unknown and excluded from these stratum IoUs. Nonpermanent is a proxy, not independently confirmed flood water. Most labelled pixels have JRC code 0, so this known-code split cannot establish a complete permanent-versus-flood water comparison. These are binary IoUs within spatial strata, not full-image multiclass flood-type IoUs.

JRC pixel counts by code: {"0": 65886206, "2": 2822300, "1": 785859, "3": 8792679}.
- permanent: scene-mean IoU 0.99751; n=27; scene-bootstrap CI [0.994613863041197, 0.9995048042461155].
- nonpermanent: scene-mean IoU 0.82203; n=28; scene-bootstrap CI [0.7586067461798516, 0.8778227035712103].
- all: scene-mean IoU 0.76622; n=33; scene-bootstrap CI [0.6977750307694166, 0.8243175764005567].

Brightness is sqrt(B2 squared+B3 squared+B4 squared)/10000, following official ml4floods get_brightness at commit 192426c7286315b59a2f52c7a9cdebe589242de2. Counts at 0.3/0.35 are saved in brightness_census.json. No arbitrary DN range is called thin cloud.

The following scenes remain separate. Their intervals use 5,000 within-scene tile-bootstrap draws and are descriptive, not event-level uncertainty:

- EMSR260_07CAMPOGALLIANO_DEL_MONIT01_v1: 4 tiles; IoU 0.19792; interval [0.0, 0.43846153846153846].
- EMSR286_09ITUANGOSOUTH_DEL_MONIT02_v1: 25 tiles; IoU 0.72037; interval [0.3091348449185786, 0.7989451716324231].

## Reproduction and limitations

- The released WF2 and Prithvi checkpoints stayed frozen for this audit; no audit threshold tuning or replacement of committed JSON occurred. A separate authorized candidate was subsequently trained for six epochs and selected/calibrated on validation data only, as recorded in `finetune_v1/`. It missed the 90% test-IoU target and reduced cloudy-test IoU. Model improvement is stopped; that candidate does not replace any frozen audit result.
- Prithvi uses six official bands, reflectance scaling and Sen1Floods11 means/stds, native 256px inputs, no metadata as configured, and two-class argmax.
- Prithvi was trained on Sen1Floods11 and is evaluated across datasets; differences cannot be attributed solely to cloud robustness.
- Prithvi runs in isolated Python 3.11 with TerraTorch 0.99.8 because the current Colab Python 3.13 runtime conflicts with its old NumPy pin. Exact installed packages are included.
- New intervals are scene-bootstrap intervals except explicitly labelled within-scene checks; they are not activation/event-clustered.
- Rule files and standalone follow-up scripts are included. Set WF_ROOT and WF_CKPT to existing downloaded inputs before rerunning.
- Source commit: fab107d0a544955301ade6fa2e900447269f57b6
- Dataset revision: 1f3faa2989e69930ac31d5c0a79fd224461123f8
- Prithvi revision: 91ce9d38086a80b078a192b374df758b8855b732

## Primary sources

- https://github.com/KanakMalpani/Reproducing-research-
- https://huggingface.co/datasets/isp-uv-es/WorldFloodsv2
- https://huggingface.co/isp-uv-es/ml4floods
- https://www.nature.com/articles/s41598-023-47595-7
- https://huggingface.co/ibm-nasa-geospatial/Prithvi-EO-2.0-300M-TL-Sen1Floods11


## Descriptive paired model accuracy

Unweighted scene means of Prithvi minus WF2 IoU, points; 5,000 scene-bootstrap draws seed 42. Gate: at least 200 GT-water pixels per scene/group. No tuning.

- val_test/all: -5.898772959894051 points; CI [-11.039671311877484, -1.4201558036081283]; n=33.
- val_test/clear: -6.254473066997678 points; CI [-12.06356465636519, -1.2679073656094058]; n=32.
- val_test/cloudy: -15.2168790085341 points; CI [-29.103670523277284, -3.135720262173285]; n=8.
- val/all: -6.545567678036232 points; CI [-16.225892938022213, 1.7774723594793578]; n=15.
- val/clear: -5.988145185908343 points; CI [-15.71187048251037, 2.2892512956541218]; n=15.
- val/cloudy: -8.577032041216487 points; CI [-9.511139246853329, -7.533390429460692]; n=3.
- test/all: -5.359777361442234 points; CI [-9.883572165763796, -1.2864336063656687]; n=18.
- test/clear: -6.48946825619415 points; CI [-13.471181776403318, -1.194273724028915]; n=17.
- test/cloudy: -19.200787188924668 points; CI [-38.74432297576346, 2.0485960463737465]; n=5.

## Corrected-rule activation-cluster follow-up (2026-11-15)

The existing corrected-paste/placebo JSON has now been activation-cluster bootstrapped in Colab, retaining all stored point estimates and scene intervals. These are new, separately labelled intervals, not replacements for any original printed paste/event quantity. The combined WF2 paired contrast uses32 scenes in24 activation clusters (including non-EMSR singleton clusters) and has interval[1.4415350645592608,5.984003104125222]; its17-scene,11-activation test-only interval[-0.38313817636896463,5.5367285405207305] includes zero. Full results for both frozen models, validation/test splits and all three contrasts are in EVENT_BOOTSTRAP_FOLLOWUP.md and corrected_paste_event_cluster_2026_11_15.json. No new point-estimate mean, model inference, training, or event-weighted point was computed. The qualitative image remains blocked by absent raw tiles in the fresh Colab runtime.
