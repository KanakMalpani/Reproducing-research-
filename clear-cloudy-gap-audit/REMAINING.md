# Remaining steps

These steps are not done. Write every new JSON file under `clear-cloudy-gap-audit/results/`. Do not replace the JSON already in this folder. Do not invent IoU numbers, scene counts, or intervals. Do not edit the paper. If a download or a path is missing, stop.

1. On Kaggle or Colab, this folder is the only code. Set `WF_ROOT` and `WF_CKPT`. Download validation and test from `isp-uv-es/WorldFloodsv2`, and `model.pt` from `isp-uv-es/ml4floods` at `models/WF2_unetv2_all`.

2. Reproduce the JSON already in this folder before any new number: the kill test, the NDVI co-match, and the hard paste.

3. Non-cloud paste placebo, and a re-run of the frozen spectrum under the written pooled-median rule. Seed 42.

4. Permanent-water versus flood-water IoU using `PERMANENTWATERJRC`, and a brightness census at 0.3 and 0.35.

5. If the tiles exist, measure `EMSR260` Campo Galliano and `EMSR286` Ituango South as their own scenes. Do not fold them into 39.3 or 5.6.

6. After the steps above, run `ibm-nasa-geospatial/Prithvi-EO-2.0-300M-TL-Sen1Floods11` on the same protocol.

7. Frozen means no extra training of `WF2_unetv2_all`. The paste stays a constructed fill.
<!-- EXECUTION STATUS 2026-11-15 -->

Execution update: see [REMAINING_STATUS.md](REMAINING_STATUS.md) for item-by-item evidence. Items1–3 and5–7 are complete. Item4 has completed brightness/JRC-supported analysis; a complete genuine flood split remains blocked by missing independent labels for unknown JRC regions. Model improvement is stopped.
