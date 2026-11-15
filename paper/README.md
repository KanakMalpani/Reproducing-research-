# CVPR-format paper source

`main.tex` is newly assembled source for the requested paper. The historical `paper/main.tex` was absent from the supplied repository; this is not claimed to be a recovered or edited historical manuscript. The author block is anonymous review metadata, not an invented author or affiliation. No PDF was required as an input. The source does not change the enrolled OpenReview abstract.

The source keeps printed frozen WorldFloods quantities unchanged, names composition/same-tile/aggregation controls, includes corrected placebo and activation intervals separately, distinguishes Prithvi and the failed candidate, and quotes the brightness census without naming a thin-cloud class. C2S-MS results are inserted only from the measured output JSON after execution.

Official style files `cvpr.sty` and `ieeenat_fullname.bst` are unchanged copies from https://github.com/cvpr-org/author-kit at commit `a2369342011145be8c85e5ad21a9b61f9fa53473` (CVPR2027 author kit, retrieved2026-11-15). Their original notices are preserved. The template choice does not claim submission, acceptance, or a reserved conference paper identifier.

Build from this directory with a TeX installation containing the official style dependencies:

```sh
pdflatex main
bibtex main
pdflatex main
pdflatex main
```

The optional WorldFloods real-tile figure remains absent because its rasters are missing. No substitute figure is presented. Genuine flood-versus-permanent recognition remains blocked by missing independent labels for JRC0 pixels. Review `clear-cloudy-gap-audit/REMAINING.md` for those open boundaries.

Validation: compiled successfully with official portable Tectonic 0.17.0, bibliography resolved, four PDF pages rendered and visually inspected. Times-compatible fonts are embedded through T1 encoding. The bundled `lineno.sty` emits a legacy source-encoding warning; no replacement characters appear in the PDF. No numerical computation was performed by the compiler. The review watermark is supplied by the official style and does not mean the paper has been submitted.
