# Corrected-paste activation-cluster follow-up

Executed in Google Colab on 2026-11-15 using Python 3.13.16 and NumPy 2.4.6. JSON-only analysis: no model inference, training, imagery download or weight changes. Both input JSON files passed before/after SHA256 checks. Every stored scene-bootstrap interval was reproduced before the new result file was written.

The point estimates below are copied verbatim from the existing summaries. No new scene-mean or event-weighted point estimate is reported. New intervals resample whole EMSR activations (non-EMSR scenes are singleton clusters), with replacement,5000 draws,seed42, percentile2.5/97.5. Each draw retains every scene of each sampled activation.

These corrected-rule intervals are distinct from the printed original hard paste5.6 [3.0,8.5], test3.8 [1.5,6.9], original event interval[3.1,8.5] and original event-weighted6.0. None of those original quantities was recomputed or replaced. All stored scene-bootstrap intervals remain intact.

| Model | Split | Statistic | Scenes / clusters | Stored mean (points) | Stored scene CI | New activation-cluster CI |
|---|---|---|---:|---:|---|---|
| WF2_UNet | val_test | gap_cloud_pts | 32 / 24 | 5.401824562730654 | [3.0395970810675137, 8.131108851983626] | [3.075068940840618, 8.185912766151164] |
| WF2_UNet | val_test | gap_placebo_pts | 32 / 24 | 1.875462012891159 | [0.7947312280281873, 3.0859464397899936] | [0.8578320052262586, 2.825463389330167] |
| WF2_UNet | val_test | cloud_minus_placebo_pts | 32 / 24 | 3.5263625498394946 | [1.4732143046767636, 5.72402744697899] | [1.4415350645592608, 5.984003104125222] |
| WF2_UNet | val | gap_cloud_pts | 15 / 13 | 6.487147269718036 | [2.943540377089716, 10.089569003886693] | [2.8600579506522243, 10.314043187560811] |
| WF2_UNet | val | gap_placebo_pts | 15 / 13 | 1.363475977077431 | [0.07172685021986815, 2.7249512104839693] | [0.015133459907284324, 2.5559841767224207] |
| WF2_UNet | val | cloud_minus_placebo_pts | 15 / 13 | 5.123671292640606 | [2.1676402497145424, 8.239488780857338] | [2.0699164028581816, 8.547435665409953] |
| WF2_UNet | test | gap_cloud_pts | 17 / 11 | 4.444186880094727 | [1.6715664311563256, 8.530741356636542] | [1.7629735767540933, 8.48763821902546] |
| WF2_UNet | test | gap_placebo_pts | 17 / 11 | 2.3272143974326833 | [0.7203564862910251, 4.1754992673787115] | [0.8586044477051036, 3.7938914642936723] |
| WF2_UNet | test | cloud_minus_placebo_pts | 17 / 11 | 2.1169724826620437 | [-0.3760086187473995, 5.187757191515272] | [-0.38313817636896463, 5.5367285405207305] |
| Prithvi_Sen1Floods11 | val_test | gap_cloud_pts | 32 / 24 | 1.48590782566094 | [0.4696010095725278, 2.557471004375826] | [0.4381431424840803, 2.567300236473625] |
| Prithvi_Sen1Floods11 | val_test | gap_placebo_pts | 32 / 24 | 2.6482709365569335 | [1.8216041771839329, 3.552474710682945] | [1.8182504010485954, 3.482509904674464] |
| Prithvi_Sen1Floods11 | val_test | cloud_minus_placebo_pts | 32 / 24 | -1.1623631108959933 | [-2.2004243456862635, -0.15842118822902607] | [-2.2810409443819806, -0.09384501825098725] |
| Prithvi_Sen1Floods11 | val | gap_cloud_pts | 15 / 13 | 1.1903907340651296 | [-0.5163359028575772, 2.783117705234394] | [-0.5994160805628322, 2.7657223642295983] |
| Prithvi_Sen1Floods11 | val | gap_placebo_pts | 15 / 13 | 2.7917169159318913 | [1.5176708486903525, 4.109407018092079] | [1.5322593415465233, 4.229170731480474] |
| Prithvi_Sen1Floods11 | val | cloud_minus_placebo_pts | 15 / 13 | -1.6013261818667626 | [-3.423324123740324, 0.17406009074473885] | [-3.617012450972485, 0.13842277829705535] |
| Prithvi_Sen1Floods11 | test | gap_cloud_pts | 17 / 11 | 1.7466582005984197 | [0.6032659401982077, 3.2256792473749116] | [0.5432280657644778, 3.1987667236368367] |
| Prithvi_Sen1Floods11 | test | gap_placebo_pts | 17 / 11 | 2.521700954755499 | [1.4933153892050481, 3.852256809807996] | [1.4742079008856361, 3.5152387892620003] |
| Prithvi_Sen1Floods11 | test | cloud_minus_placebo_pts | 17 / 11 | -0.7750427541570797 | [-1.785079626910426, 0.1376059740370137] | [-1.9813404199835527, 0.39864262910528836] |

For corrected WF2, the combined paired cloud-minus-placebo interval excludes zero; the test-only paired interval includes zero. For Prithvi, the combined paired contrast is negative: the noncloud placebo drop exceeds the cloud-fill drop. Its test-only paired interval includes zero. A smaller cloud-fill drop is not a better-model claim.

Both fills are constructed constant fills on clear tiles, scored only on unpainted valid pixels. They are not natural cloud. No new decision bar or causal interpretation is introduced.

Source rule: CORRECTED_PASTE_EVENT_RULE.md. Executable script: corrected_paste_event_cluster.py. Full precision, cluster membership and input hashes: corrected_paste_event_cluster_2026_11_15.json.

Qualitative figure status: stopped at the missing-input boundary. A preflight in the reconnected Colab runtime found /content/Reproducing-research-/clear-cloudy-gap-audit, /content/worldfloods and /content/wf-models/model.pt all absent. Only the four small saved JSON/script/rule files were uploaded into a fresh, separate follow-up folder for this analysis. No rasters were recreated or downloaded and no substitute imagery was used.
