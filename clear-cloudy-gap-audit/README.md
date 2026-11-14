# What Does the Clear–Cloudy Gap Measure? A Controlled Audit of WorldFloods Flood Segmentation

Branch: `cursor/fix1-wording-revision-cf46`

Python: 3.12.3. The committed cluster run imported NumPy 2.5.4.

This folder is the reproduction point for the activation-clustered intervals. The three analysis files here are byte copies of the committed files under `revision/`. They were copied, not retyped. SHA-256:

- `EVENT_CLUSTER_RULE.md` `e7154e29d1e9b8e279d1f69c787c73fc83908bacfdd35b141690e8fcda578b0b`
- `event_cluster.py` `407ed313fac83184e52bb478b54411d4c690752cb254c5d41845504743db3a0a`
- `event_cluster_results.json` `23e1c3c63008fc38dbd2f883712af1d0dfc40350f8b266c54a2a46095b6b1245`

## Command

From the repository root:

```bash
python3 revision/analysis/event_cluster.py
```

That command reads the committed JSON under `topics/worldfloods-cloud-aware/` and writes `revision/analysis/event_cluster_results.json` and `revision/supp/event_cluster_auto.tex`. It checks each scene-bootstrap interval against the committed files before it writes. It does not load tiles or a checkpoint.

Run that path. Do not run `clear-cloudy-gap-audit/event_cluster.py`. The script finds the repository root from its location under `revision/analysis/`.

## Files in this folder

- `README.md`
- `REQUIREMENTS`
- `CITE_AND_RUN.md`
- `EVENT_CLUSTER_RULE.md`
- `event_cluster.py`
- `event_cluster_results.json`
