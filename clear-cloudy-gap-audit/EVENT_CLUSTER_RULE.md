# Event-clustered intervals

Locked before any clustered interval is computed. This rule re-reads committed JSON. It does not load imagery, run the UNet, or change a locked bar, seed, or scene-mean statistic.

The paste stays a constructed fill on clear tiles, scored on unpainted pixels. Nothing in this rule calls that fill natural cloud.

## Cluster

The primary cluster is the Copernicus EMS activation code. For a scene id, take the leading `EMSR` and the digits that follow it (`EMSR347_06MWANZA_DEL_v1` is cluster `EMSR347`). A scene id that does not start that way, including `RS2` and `ST1`, is its own cluster.

Scenes that share an activation stay together when a subset is taken. A scene that is not in the subset is left out, so a cluster can shrink. A scene with a non-finite gap for that statistic is left out.

## Statistics

The locked statistic is the unweighted mean of the scene gaps that enter the statistic, in the order of the committed file. The interval is a cluster bootstrap: 5000 draws, percentile 2.5 and 97.5, `numpy.percentile` with its default linear method, the same call the existing scene bootstrap uses.

A draw resamples clusters with replacement, one draw for each cluster in the statistic. Every scene in a drawn cluster is included, and a cluster drawn twice is included twice. The draw's value is the unweighted mean of that scene list.

Each statistic uses a fresh `numpy.random.default_rng`. Paste-family statistics use seed 42. Match statistics use seed 0. The index draw `rng.choice(n_clusters, n_clusters, replace=True)` is the resample. On this environment's NumPy it matches `rng.choice(gaps, n)` when every cluster is one scene. The run must check that identity and must also recompute the ordinary scene bootstrap and match the committed interval. If either check fails, the script writes no table.

Paste family, from the committed rows, finite gaps only:

| Statistic | File | Field |
| --- | --- | --- |
| Hard constant paste, all / validation / test | `p2_results.json` | `gap_pts`, split as stored |
| Own mask and spectrum | `p2_results.json` | `gap_pts` where `own_mask` and `own_spectrum` |
| Borrowed or frozen | `p2_results.json` | the complement |
| Shuffled 16 px blocks | `p2_shuffled_results.json` | `gap_sh` |
| Real minus shuffled | `p2_shuffled_results.json` | `gap_real - gap_sh` |
| Hard paste, distance bins | `p2_shuffled_results.json` | `gap_real_bin0`, `gap_real_bin1`, `gap_real_bin2` |
| Soft edge | `p2_soft_edge_results.json` | `gap_soft` |
| Per-pixel spectrum behind the soft edge | `p2_varying_results.json` | `gap_soft_var` |

Match family, using the gaps already stored, not gaps recomputed from tiles:

| Statistic | File | Field |
| --- | --- | --- |
| Unmatched clear versus cloudy | `kill_test_results.json` | `test_val.per_scene[].gap` |
| Water-fraction match | `method1_results.json` | `terrain.matched[].gap` |
| NDVI co-match | `ndvi_comatch_results.json` | `ndvi_comatch.per_scene[].gap` |

The secondary point is the event-weighted mean: the unweighted mean of the within-cluster means. Its interval resamples clusters the same way, and the draw's value is the mean of the drawn clusters' means.

## What the numbers may change

The scene-bootstrap intervals already in the paper stay. This rule does not re-score them and does not move the bar of 5.

- If the activation-clustered lower bound for the 32-scene hard paste is at most 0, the paper says the scene mean is positive under scene resampling and not distinguishable from 0 under event resampling.
- Otherwise the paper may say that activation clustering leaves that conclusion unchanged, and it prints the measured interval.
- The published hard-paste scene mean minus the published 32-scene pixel-pool drop of 2.8 is the margin. If the scene mean exceeds the event-weighted mean by at least that margin, the aggregator caveat moves into the claim sentence. A smaller difference leaves the caveat where it is.

No other statistic gets a new bar.

## Sensitivity grouping

A second grouping may merge primary clusters. It is not the primary interval.

Fetch `https://rapidmapping.emergency.copernicus.eu/backend/dashboard-api/public-activations/?code=EMSR<code>` for every activation code in these tables. Merge two codes only if a non-empty `gdacsId` is identical, or if the `relatedevents` field of either record contains the other code as an `EMSR` plus digits. Same country, same year, or a similar title is not a merge. Non-EMSR scenes are never merged.

If any request fails, the payload is not JSON, the response exceeds 8 MB, or those fields are absent, publish no sensitivity interval and record the reason. Do not type a merge by hand.

## Match accounting

Replay the water-fraction matcher in `method1_analyze.py` and the NDVI matcher in `ndvi_comatch_analyze.py` on `method1_tiles.json`. Do not change the tolerances, the greedy order, or the 200-pixel gate.

Every scene that passes the unmatched gate is a row, including a scene with no matched pair. For the water-fraction match, cloudy tiles available are tiles with cloud fraction at least 0.1, matched tiles are those assigned a clear partner with absolute water-fraction difference at most 0.05, and discarded tiles are available minus matched. Clear tiles used are the distinct clear partners.

For the NDVI co-match, a cloudy tile with no NDVI is skipped and counted separately. Matched tiles meet both absolute differences of 0.05. Discarded tiles are the cloudy tiles that have an NDVI and were not matched.

The replay must reproduce the stored pair counts and the stored gaps. If it does not, publish no accounting table.

## Display

Full precision stays in the JSON. The table shows one decimal, half up. If that rounding would move a bound across 0 or across 5, that bound is printed to three decimals.

## After a failed check

Publish nothing. A later commit may fix a reader that pointed at the wrong field. It may not change the cluster definition, the seeds, the draw count, the statistics, or the reading rules above.
