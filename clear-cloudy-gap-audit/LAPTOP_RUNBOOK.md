# Kaggle / Colab runbook: finish the WorldFloods clear–cloudy audit

An agent on Kaggle or Google Colab should follow this file. The agent’s workspace is this folder, `clear-cloudy-gap-audit/`. Every command below runs here. Paper: “What Does the Clear–Cloudy Gap Measure?” Branch: `cursor/fix1-wording-revision-cf46`. PR: https://github.com/KanakMalpani/IEEE-CVF/pull/3

Do not invent IoU numbers, scene counts, or intervals. If a download fails, stop and say so.

## What “frozen” means

Score the released checkpoint `WF2_unetv2_all` with no extra training. Train scenes stay out because the checkpoint saw them.

## What is already done

- Wording, three illustration figures, event-clustered intervals, and a sample-size table are on the PR.
- Committed gaps: 8 unmatched (39.3 [22.4, 57.6]), 7 water-fraction (8.9 [−0.7, 17.8]), 5 NDVI (1.4 [−24.5, 24.5]), 32 hard paste (5.6 [3.0, 8.5]).
- Tile metadata: 1291 tiles, 34 scenes (16 val, 18 test). Two scenes have tiles and no stored gap: `EMSR260_07CAMPOGALLIANO_DEL_MONIT01_v1` (4 tiles) and `EMSR286_09ITUANGOSOUTH_DEL_MONIT02_v1` (25 tiles).

## Notebook setup

Turn on a GPU (Colab T4, or a Kaggle GPU notebook). Enable internet. Kaggle disk is tighter than Colab, so download val and test only.

```python
# Colab or Kaggle, first cell
!git clone https://github.com/KanakMalpani/IEEE-CVF.git
%cd IEEE-CVF
!git fetch origin cursor/fix1-wording-revision-cf46
!git checkout cursor/fix1-wording-revision-cf46
%cd clear-cloudy-gap-audit
!pip install -q rasterio huggingface_hub
```

Torch is already installed on Colab and Kaggle GPU notebooks. After the cell above, the working directory is this folder. Scripts, locked rules, `NOTE.md`, and the committed result JSON are the files next to this runbook. Do not look for them under `topics/` or `revision/`.

## Downloads

The git repo does not contain rasters or weights. On the notebook, use `/content` (Colab) or `/kaggle/working` (Kaggle):

```bash
# pick one root
export WF_ROOT=/content/worldfloods          # Colab
# export WF_ROOT=/kaggle/working/worldfloods # Kaggle
export WF_CKPT=/content/wf-models/model.pt
```

- Dataset: Hugging Face `isp-uv-es/WorldFloodsv2` (CC BY-NC). Pull val and test only. A full mirror is also listed as `tacofoundation/worldfloods` (~76 GB). Do not download train unless a later step asks for it.
- Checkpoint: Hugging Face `isp-uv-es/ml4floods`, folder `models/WF2_unetv2_all`, file `model.pt` (13-band UNet, about 31 MB). Point `WF_CKPT` at that file.
- JRC permanent-water masks, if the dataset ships them, sit beside `gt/` as `PERMANENTWATERJRC/`. The code does `gt.replace("/gt/","/PERMANENTWATERJRC/")`.
- Second model, only after the first runs: `ibm-nasa-geospatial/Prithvi-EO-2.0-300M-TL-Sen1Floods11`

`WF_ROOT` and `WF_CKPT` are the only paths outside this folder. `wf_data.py` reads `WF_ROOT`. `cloud_iou_pilot.py` reads `WF_CKPT`. JRC masks, when the dataset has them, are inside `WF_ROOT` beside `gt/`. Read `wf_data.py` before changing `WF_ROOT`. Read each `P2_*.md` and `NDVI_COMATCH_RULE.md` in this folder before the matching script. Do not change a locked bar after seeing a number.

`val_cloud_counts.py` and `label_histogram_test.py` still contain `/home/box/worldfloods`. `m2_cache.py` and `m2_train_eval.py` still contain `/workspace/wf-cache`. If that path is missing, stop. Do not add a third root.

## Where files are

Committed JSON stays in this directory. That includes `kill_test_results.json`, `method1_results.json`, `method1_tiles.json`, `ndvi_comatch_results.json`, `p2_results.json`, `p2_soft_edge_results.json`, `p2_varying_results.json`, `p2_shuffled_results.json`, and `cloud_iou_tiles.json`. New JSON is written only to `results/`. Do not replace a committed file with a new run.

`event_cluster.py` and `sample_size.py` are copies. Do not run those two from this folder: each one finds its inputs from a `revision/analysis/` path. Their committed outputs are already here, `event_cluster_results.json` and `sample_size_results.json`.

```bash
# working directory: clear-cloudy-gap-audit/
mkdir -p results
# cloud_iou_pilot.py is the one script that takes an output prefix
OUT=results/cloud_iou python cloud_iou_pilot.py
```

Other scripts open a fixed filename in the working directory. Keep the committed file, and if the bytes change, move the new file into `results/` and put the committed file back:

```bash
# working directory: clear-cloudy-gap-audit/
ref_run () {
  out="$1"; shift
  cp "$out" "$out.ref"
  python "$@"
  if ! cmp -s "$out" "$out.ref"; then
    mv "$out" "results/$(basename "$out")"
  fi
  mv "$out.ref" "$out"
}
```

## Runs, in order

1. Confirm the checkout reproduces the stored gaps before any new number. From this folder: `ref_run kill_test_results.json kill_test.py`, then `ref_run ndvi_comatch_results.json ndvi_comatch_analyze.py`, then `ref_run p2_analyze.out.json p2_analyze.py`. A match leaves no file in `results/`. A difference is a new file under `results/` and is not a stored gap.
2. Non-cloud paste placebo and a re-run of the frozen spectrum under the written pooled-median rule (`P2_CLOUD_PAINT_RULE.md`). Same checkpoint, seed 42. Write that JSON under `results/`. Soft-edge and distance bins already exist in this folder. Do not rerun them as if they were missing.
3. Permanent water versus flood water: IoU split using `PERMANENTWATERJRC` inside `WF_ROOT`, plus a brightness census at the 2023 cuts 0.3 and 0.35 (Portalés-Julià et al.). Write the new JSON under `results/`. Do not claim 500–2000 DN is thin until this is measured.
4. Optional new scenes only if the tiles exist: EMSR260 Campo Galliano and EMSR286 Ituango South. Report them as new measurements with their own intervals under `results/`. Do not fold them into 39.3 or 5.6.
5. Second model: Prithvi Sen1Floods11 checkpoint on the same WorldFloods tiles, same protocol, only after steps 2 and 3. Its JSON goes in `results/`.
6. The new JSON is already in `results/` on this branch. Rebuild with pdfLaTeX only if the full repo is present. Keep the body ending on page 8. New tables go in the supplement if the body is full. Do not edit the paper from this folder.

## What not to write

- Do not call the paper fake, and do not present the illustration JPEGs as WorldFloods tiles.
- Do not describe the hard paste as natural cloud. Keep: the paste is a constructed fill on clear tiles, scored on unpainted pixels.
- Do not use the title “Are Clouds Really to Blame?”
- Do not add authors. OpenReview 14667: title, subject, keywords, and abstract can still change until the paper deadline. Author list cannot.
- Hypothetical n=20 and n=50 in the supplement are extrapolations. Do not replace them with a made-up mean.

## Where new numbers go

New JSON goes only in `clear-cloudy-gap-audit/results/`. The committed JSON in this directory stays as it is. Do not edit the paper from this folder. Keep this runbook at `clear-cloudy-gap-audit/LAPTOP_RUNBOOK.md`.
