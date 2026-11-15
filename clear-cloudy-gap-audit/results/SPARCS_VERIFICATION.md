# SPARCS execution and preservation checks

Completed in the user's Colab T4 runtime on 2026-11-15; no training. Run `python clear-cloudy-gap-audit/sparcs_verify_results.py` in the complete repository for read-only record checks. It performs no inference, bootstrap or estimation.

- Official archive:1,555,733,976 bytes; observed SHA256 `5cde604615ee241950b5a0b641ae5de82b27738b51e39aa724d18faec0bbaa9a`; all ZIP CRCs passed. Acquisition-named MTL files were mapped explicitly after the initial downloader stopped; no inputs were fabricated.
- Export:564,220 bytes, SHA256 `151583ae31ea0e7c1d0f4f28319c7be1977f327568227278021f974641e1145f`, identical to the Colab download receipt.
- All80 subscenes/acquisitions have original imagery/manual labels/calibration. The fixed census retained910/1280 windows, with all exclusions recorded.
- All new JSON identifies Landsat 8 SPARCS and DOI10.5066/F7FB5146. Source, rule and manifest hashes match executed inputs. Calibration coefficients and native30m geometry are retained per chip.
- Tile confusion/GT-water conservation, matching uniqueness/tolerances, actual eligible-acquisition counts, identical fill scoring-surface water totals and exact copied summaries passed.
- Frozen Prithvi checkpoint SHA256 is unchanged: `76eed77d8bd543ae441308b80e8408502243a871cb8a338ddc8220ed96dfc270`. Evaluation mode/no gradients and finite FP32 logits were enforced. No WF2 checkpoint was loaded.
- All72 previously tracked JSON files remain unchanged versus pre-results baseline `a1d44e2`, allowing only Windows/Git CRLF representation differences.
- Original numerical WorldFloods and C2S-MS manuscript sections remain unchanged. New table values round the measured SPARCS JSON; no mean or interval was recomputed locally.
- Updated manuscript compiled with Tectonic0.17.0 to four pages, rendered and visually inspected. No overfull boxes, unresolved references/citations or replacement characters remain. Bibliography resolved without warnings. The dependency's legacy lineno source-encoding warning and archive TIFF photometric warnings are documented; no input image was rewritten to suppress them.

Interpretation remains limited: different unmatched/matched samples (25/9/8), cross-sensor/native-resolution changes, and visible-water rather than hidden-water labels. All composition intervals and the paired fill interval include zero; this does not establish equivalence or a universal cloud penalty.
