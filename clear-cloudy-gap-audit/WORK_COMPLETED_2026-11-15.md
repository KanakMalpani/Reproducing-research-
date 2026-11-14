# Work completed and preservation — 2026-11-15

## Current decision

Model improvement is stopped. No second candidate was trained. Keep the original frozen WF2 checkpoint as the audit reference. The separately trained candidate is an experimental artifact; its validation gain did not demonstrate a general test or cloud-robustness improvement.

## Repository and execution

- Source: https://github.com/KanakMalpani/Reproducing-research- ; commit fab107d0a544955301ade6fa2e900447269f57b6.
- Colab notebook: https://colab.research.google.com/drive/1YrTDLg30nOGAqX7Cffwp0fkIR6qTBS1H . Scientific GPU work ran on the Colab T4.
- Dataset revision: 1f3faa2989e69930ac31d5c0a79fd224461123f8. Download manifests contain sizes/checksums and exact model revisions.
- Original audit analyses reproduced before new results: kill_test.py, ndvi_comatch_analyze.py and p2_analyze.py all matched committed JSON exactly. Frozen GPU inference matched every confusion count across1,291 tiles and34 scenes.
- Corrected pooled cloud-pixel median and noncloud placebo, permanent/nonpermanent strata, official brightness thresholds, separate scene checks and Prithvi same-protocol comparison completed. See results/RUN_REPORT.md and REMAINING_STATUS.md.
- Original paper and committed JSON were preserved. New JSON is under results/. The corrected follow-up does not replace the original published hard-paste figure.

## Separate fine-tuning experiment

388 training scenes and38,596 eligible training tiles; exclude EMSR activations found in validation/test. Seed42, full UNet fine-tune, AdamW encoder LR5e-6/decoder LR5e-5, batch8,512 steps/epoch, masked water BCE+Dice and auxiliary cloud BCE, spatial/radiometric augmentation. Six epochs completed; early stopping retained epoch2. Candidate/checkpoint and calibration selection used validation only. Threshold grid .20–.90 chose .50 with zero additional bias shift. Existing test was previously inspected for the original audit and is not a fresh blind benchmark.

| Split/group | Released water IoU | Candidate water IoU |
|---|---:|---:|
| val/all | 86.5145% | 91.6259% |
| val/clear | 86.7104% | 91.8752% |
| val/cloudy | 67.4644% | 76.6313% |
| test/all | 78.3649% | 79.2819% |
| test/clear | 79.5532% | 79.4323% |
| test/cloudy | 76.6755% | 73.9034% |

90% pooled test-water IoU target: **not met**. Paired mean test scene change: -0.4026 points;95% scene-bootstrap CI [-5.988669374462191, 3.8530393481277763]. Cloudy-test IoU declined. The candidate should not be described as generally better or more cloud robust.

Released checkpoint SHA256: `ef52e243cea1c659895d05440932b7ec0ef473f63a52283b1c9bd459bf1982b7`; verified unchanged. Candidate SHA256: `f8ed2fb50a788279e38c3f0d0c6f895aafe8cc3afb7bd018a5f608050a2bcdce`. Bare compatible candidate weights and full optimizer/training state are preserved under checkpoints/finetune_v1/. Exact configuration, all epoch results, calibration curves and candidate controls are under results/finetune_v1/.

## Remaining audit follow-up

The final frozen JRC analysis separates code1 flood proxy, code2 seasonal, code3 permanent and code0 unknown, plus clear/cloud pixel subsets and GT-water coverage. Read REMAINING_STATUS.md for measured values and the explicit missing-label limitation. No model weights changed during this follow-up.

## Stored artifacts

User-requested local storage: C:\Kanak\Reproducing-research-2026-11-15\repository. The preserved archive contains repository source and Git metadata, rules, result files, baseline and candidate checkpoints, full training state, and Prithvi model assets. A separate completion archive adds the final JRC analysis. SHA256 inventory is in results/preservation_verification.json.

Raw downloaded satellite rasters (about71GiB including training) remain outside the repository in Colab /content/worldfloods and are not included in the local archive. Rebuildable Python virtual environments are also excluded. Dataset/checksum manifests and environment requirements are retained. Colab runtime storage is temporary: the manifests allow retrieval, but do not themselves preserve the raw bytes. No claim is made that all raw rasters were stored locally.

At the end of the preservation phase, no GitHub push or publication had been performed. Subsequent repository publication is recorded in Git history. See MODEL_ARTIFACTS.md for which checkpoints are included in Git and which remain in local storage. The paper is untouched.
