# Work log and handover

Model improvement stopped at the user request on 2026-11-15. No second candidate was trained. Released WF2 weights remain frozen.

## Frozen audit
Reproductions matched committed JSON and GPU confusion counts. Corrected pooled medians, noncloud placebo, JRC known-code strata, brightness .3/.35, separate Campo Galliano and Ituango South, and frozen Prithvi comparison completed. See results/RUN_REPORT.md and result JSON.

## Separate authorized fine-tune
Six epochs; epoch 2 selected by validation only. Train388 scenes, 38596 tiles, excluding evaluation EMSR activations. Validation water IoU 0.8651451103384014 to 0.9162587023504096. Test pooled water IoU 0.7836493970690738 to 0.7928193855368155. 90% test target not met. Cloudy test IoU decreased from 0.7667548616804217 to 0.7390338550118456. Paired mean scene change -0.4026345468267597 points, 95% CI [-5.988669374462191,3.8530393481277763]. This does not establish a general improvement.
Threshold calibration considered .20 through .90 on validation only; selected .50, bias shift zero. Weights and full optimizer/training state preserved. Existing test was previously inspected, not a fresh blind benchmark.

## Remaining audit limitation
JRC code0 has no observations; previous known nonpermanent {1,2} metric is not confirmed flood water. A completion follow-up must report historically nonwater code1 separately from seasonal code2 and quantify coverage. No unknown pixel should silently become flood.

## Preservation
Archive contains repository source, rules, result JSON, model assets, trained checkpoints and optimizer state. External raw rasters and rebuildable .venv caches are not part of this repository archive; dataset/checksum manifests and environment requirements preserve retrieval information.

## Final handover update

The frozen JRC completion follow-up is now finished. Read WORK_COMPLETED_2026-11-15.md and REMAINING_STATUS.md for the final measured coverage, known-category results, blocked genuine-flood split, and stopped model-improvement status.
