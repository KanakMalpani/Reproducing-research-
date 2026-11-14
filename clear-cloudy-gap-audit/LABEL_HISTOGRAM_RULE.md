# Test label histogram: locked rule (committed before running)

- Split: **test** only (val census already in val_cloud_counts.json). All test scenes with readable S2 + gt.
- Pixels: labelled (gt water band > 0). Groups: GT-cloud (gt band0 == 2) vs GT-clear (band0 == 1).
- Histograms: s2cloudless probability (S2 band 15, 0-100) in 10-point bins; visible brightness = mean of raw B2,B3,B4
  (bands 2-4, DN) in 500-DN bins up to 10000 (+overflow).
- Report: per-group counts, and fraction of GT-cloud pixels with s2cloudless < 50 and < 20 ("thin inside GT-cloud"),
  fraction of GT-clear pixels with s2cloudless >= 20 ("thin labelled clear"). Thin-class verdict: GT-cloud contains a thin
  class iff >= 5% of GT-cloud pixels have s2cloudless < 50.
