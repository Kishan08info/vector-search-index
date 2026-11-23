# Vector search benchmark — median across 3 full re-runs (seed 42)

Recall, topic precision and docs scanned are identical in every run (deterministic given the seed). Latency is the median of 7 reps per run, then the median across the 3 runs; the range in brackets is the min–max of the 3 run medians. Timings were measured on a shared-CPU VM and vary run to run — treat speedups as approximate, recall as exact.

| nprobe | recall@10 | topic P@5 | ms/query (median of 3) | speedup (median of 3) | docs scanned |
|---:|---:|---:|---:|---:|---:|
| 1 | 0.8300 | 0.9840 | 1.01 [0.94–1.04] | 2.82x | 1862 (1.9%) |
| 4 | 0.9650 | 0.9530 | 1.05 [1.02–1.37] | 2.48x | 6503 (6.5%) |
| 8 | 0.9845 | 0.9390 | 1.52 [1.49–4.38] | 1.72x | 12512 (12.5%) |
| 16 | 0.9945 | 0.9320 | 1.97 [1.91–1.99] | 1.36x | 24930 (24.9%) |
| 32 | 0.9995 | 0.9280 | 2.60 [2.59–3.26] | 1.01x | 49757 (49.8%) |

Brute force: 2.66 ms/query (median of 3 run medians), topic precision@5 0.928, source doc in top-10 150/150.
