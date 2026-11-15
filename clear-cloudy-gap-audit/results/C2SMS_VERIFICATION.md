# Verification of the completed C2S-MS export

Run `python clear-cloudy-gap-audit/c2sms_verify_results.py` from a complete Git checkout. This reads stored results and checks records; it performs no inference, training, bootstrap or estimation. It uses the fixed pre-results commit `e79db6acacae62c405ccd3daf2d1d12ea343bc41` to verify all 65 previously tracked JSON files, allowing only Git CRLF/LF representation differences.

Passed on 2026-11-15:

- Download archive size and SHA256 matched the value printed by Colab: 2,628,975 bytes, `6966eb5897ff20a5cd7bbe9d39d4239f532f7873413a256c788ee0b352d4dcf4`.
- All 8,100 required download records were complete. The preflight rule and inventory hashes agree with the exact cloud inputs; Windows inventory line endings normalize to those same bytes.
- All required and supporting C2S-MS JSON files identify the dataset and DOI.
- All cloud-executed script hashes agree with the preserved code. The added verification script is a post-run record checker, not an analysis change.
- Every tile's full confusion equals clear plus cloud confusion, with GT-water totals and valid-pixel bounds conserved.
- Matching has no repeated clear/cloudy tile within an event, obeys both fixed tolerances, and agrees with the recorded pair and eligible-event counts.
- Original, cloud and land fills have identical common-surface GT-water totals in each event; the paired subtraction and copied aggregator summaries agree exactly.
- Prithvi is recorded frozen, no training, with the checkpoint unchanged after execution. WF2 was not loaded by the wider-data script.
- All 65 pre-existing tracked JSON files match the pre-results commit. No original result was replaced.
- Every requested long-precision paper number and brightness count is present; the three composition-table rounded values match the C2S-MS output. Official CVPR style files are byte-identical to author-kit commit `a2369342011145be8c85e5ad21a9b61f9fa53473`.
- The actual manuscript compiled with Tectonic 0.17.0 to four PDF pages. All pages were rendered and visually inspected. No overfull boxes, unresolved citations/references, or replacement characters remain. The dependency's legacy `lineno.sty` encoding warning does not appear as a rendered defect.

No intervals or means were re-estimated during these local checks. All experimental inference and bootstrap calculations ran in Colab. The optional WorldFloods illustration and genuine flood split remain unresolved for the stated missing inputs.
