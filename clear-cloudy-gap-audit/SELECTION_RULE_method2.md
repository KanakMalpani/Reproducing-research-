# Selection rule: method 2, a cloud-aware output adapter on a frozen WF2 UNet (committed before training, 2026-11-03)

Frozen: all of WF2_unetv2_all, encoder and decoder. Trained: a small residual adapter A on per-pixel inputs [water logit, cloud logit, s2cloudless cp] (or without cp, as an ablation).
It outputs a delta added to the water logit: water = sigmoid(lw + A(.)) > 0.5. The last conv is zero-initialized, so it starts at the baseline.
Architecture: conv3x3(in,h)-ReLU-conv3x3(h,h)-ReLU-conv1x1(h,1). Train tiles only (256 px, 512 stride, >=50% valid).
Loss: BCE on valid water pixels (labels 1/2), pos_weight 1, Adam lr 1e-3, 4 epochs, batch 8, seed 0. 3 CPU threads.
Configs (4): h in {8,16} x cp-input in {yes,no}.
Selection (val only): among configs whose val clear-tile (cf<0.01) pooled water IoU is >= baseline - 1.0, take the max val pooled cloudy-tile (cf>=0.1) IoU. If none beats baseline on val, declare FAIL without opening test.
Test, scored once against the NOTE.md bar: cloudy-tile IoU delta >= +5 pts with the 95% scene-bootstrap CI excluding 0, and clear-tile loss <= 1 pt.
Also reported on test, for both baseline and selected (primary reporting metric):
- the terrain-matched gap (method1_analyze matching: same scene, |water-frac diff| <= 0.05)
- the terrain-matched cloudy-tile IoU
Exploratory (dev only, never scored on test): widen the method-1 threshold grid to a in {0.5,0.6,0.8,1.0,1.5} on val.
