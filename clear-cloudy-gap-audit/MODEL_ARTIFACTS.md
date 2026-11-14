# Model artifacts

The selected experimental candidate is included at `checkpoints/finetune_v1/calibrated_selected_model.pt` (31,165,457 bytes). SHA256: `f8ed2fb50a788279e38c3f0d0c6f895aafe8cc3afb7bd018a5f608050a2bcdce`.

This is a bare PyTorch state dictionary with the original `network.` key prefix, compatible with the audit's UNet loader. The water-output threshold remains 0.5. This candidate achieved 91.63% validation water IoU and 79.28% test water IoU. Cloudy-test performance declined; it is an experimental artifact, not a replacement for the frozen research baseline. Model improvement is stopped.

The complete local preservation copy is `C:\Kanak\Reproducing-research-2026-11-15\repository`. It includes the original baseline, duplicate selected/unconstrained training checkpoints, the full optimizer/training state and the Prithvi checkpoint. These additional binary files are excluded from Git. Their sizes and SHA256 hashes are recorded in `results/preservation_verification.json`.

The original public checkpoints can be retrieved using the pinned revisions and checksums in `results/download_manifest.json` and `results/prithvi_download_manifest.json`, with `download_inputs.py` and the asset-download procedure recorded in `notebooks/WorldFloods_controlled_audit.ipynb`. Prithvi's configuration and requirements are included in `prithvi_assets/`; the approximately 1.28 GB checkpoint is not included in Git.

Raw satellite rasters and virtual environments are excluded. The input manifests preserve retrieval information, not the raw bytes. See `WORK_COMPLETED_2026-11-15.md` and `REMAINING_STATUS.md` for the complete results and the unresolved independent flood-label limitation.
