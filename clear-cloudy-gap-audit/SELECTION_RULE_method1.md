# Selection rule: cloud-adaptive water threshold (committed before any method run, 2026-11-03)

Model: frozen WF2_unetv2_all (no retraining). Water prob p_w, s2cloudless band 15 cp in [0,1], model cloud head p_c.
Candidates (fixed grid, 17 configs incl. baseline):
- T (threshold): water = p_w > clip(0.5 + a*cp, 0.05, 0.95), a in {-0.4,-0.3,-0.2,-0.1,0.1,0.2,0.3,0.4}
- G (cloud-head gate): water = p_w * (1 - b*p_c) > 0.5... evaluated as p_w > 0.5 / max(1 - b*p_c, 1e-3)·, b in {-0.4,-0.3,-0.2,-0.1,0.1,0.2,0.3,0.4}
- baseline: a = b = 0
Context, seen before writing this rule (tile JSON from the earlier pilot): errors in cloudy tiles are mostly false positives (FP 284k vs FN 76k), so both signs are in the grid.
Selection (val only): among configs whose val clear-tile (cloud frac < 0.01) pooled water IoU is within 1.0 pt of baseline, pick the max val pooled water IoU on cloudy tiles (cloud frac >= 0.1). Ties go to the smaller |a| or |b|. If nothing beats baseline on val, report the method as failed and skip test.
Test (scored once): delta = selected - baseline on test.
PASS iff the cloudy-tile pooled IoU delta is >= +5 pts, its 95% scene-bootstrap CI (5000 draws) excludes 0, and the clear-tile IoU loss is <= 1 pt.
Same tiling as cloud_iou_pilot.py (256 px tiles, 512 stride, >=50% valid).
