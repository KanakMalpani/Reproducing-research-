# P2 follow-up: IoU drop by distance to painted mask (locked before computing)

Status: FOLLOW-UP, descriptive; computed in the shuffled-control pass for both real and shuffled masks.
- Per unpainted pixel: Euclidean distance (scipy.ndimage.distance_transform_edt) to the nearest painted pixel.
- Bins (0,8], (8,32], >32 px. Per scene pooled confusion per bin; gap = (orig - painted IoU) x 100.
- Scene mean + 95% bootstrap CI (5000, rng 42) per bin over scenes with finite IoU in that bin.
- Spillover reading: drop concentrated in (0,8] and ~0 beyond 32 px. Global paste-shift reading: flat across bins.
