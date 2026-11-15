# Third-dataset stress test: SPARCS

This follow-up was requested after the differing WorldFloods/C2S-MS findings. It holds the released Prithvi checkpoint fixed and adds **Landsat 8 SPARCS**, DOI10.5066/F7FB5146. This is a cross-sensor, visible-water stress test; it does not isolate dataset shift from sensor/resolution/label changes, establish hidden-water accuracy, or claim a new flood-event benchmark.

The protocol was published at `b43794d` before download or label census. The first complete reader/evaluation implementation was published at `a1d44e2` before execution. Neither commit contains measured SPARCS outcomes. All existing result JSON is immutable.

## Dataset selection

- [USGS SPARCS](https://www.usgs.gov/landsat-missions/spatial-procedures-automated-removal-cloud-and-shadow-sparcs-validation-data) supplies separate manual land, water, flooded-area, cloud, shadow and snow classes and original Landsat imagery/MTL calibration. It supports explicit visible-water scoring without deriving a cloud label from brightness.
- [S1S2-Water](https://github.com/MWieland/s1s2_water) merges cloud, shadow and no-data in its valid mask; those invalid pixels cannot all be relabelled cloud.
- [STTORM-CD-Floods](https://zenodo.org/records/14891438) has a separate cloud channel but its binary labels identify change, not total surface water. No change mask was passed off as a water mask.
- [SEN12-FLOOD](https://doi.org/10.21227/w6xz-s898) has image-level flooding flags, which do not provide the pixel water labels required for this IoU protocol.

## Colab execution

Use the already verified C2S-MS Colab environment and pinned frozen assets at `/content/c2sms_run/clear-cloudy-gap-audit/`, with its `.venv-prithvi-c2sms/bin/python`, strict checkpoint/config checksums and original `c2sms_controls.py`. That helper's SHA256 is verified before importing its fixed matching/bootstrap functions. This is code reuse only: no C2S-MS imagery, fill spectrum, cloud label, water label or prediction enters SPARCS scoring.

Create a separate `/content/sparcs_run/clear-cloudy-gap-audit/` folder containing the locked rule and the SPARCS scripts. Run in order:

```python
subprocess.run([python, '-u', str(audit/'sparcs_download.py')], check=True)
subprocess.run([python, '-u', str(audit/'sparcs_data.py')], check=True)
# Inspect geometry, calibration availability, label coverage and eligibility first.
subprocess.run([python, '-u', str(audit/'sparcs_prithvi_evaluate.py')], check=True)
```

Here `python` names the existing isolated interpreter and `audit` the new SPARCS folder. The download and preflight create new JSON exclusively; do not bypass their stop conditions or rerun over completed output. Export the new results, runtime/environment provenance and exact executed scripts. Record actual eligible counts and intervals only after successful execution.

The [USGS calibration formula](https://www.usgs.gov/landsat-missions/using-usgs-landsat-level-1-data-product) is applied to original DN before model normalization. Native 30m imagery is not resampled to pretend it is Sentinel-2. No training, sensor harmonization or threshold fitting is allowed. Full operational differences and limitations are fixed in `SPARCS_FROZEN_CONTROL_RULE.md`.
