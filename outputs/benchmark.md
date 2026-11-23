# Vector search benchmark (synthetic data, seed 42)

Corpus: 100000 docs, 128 dims, 50 topics, 200 queries.
Brute force: 2.232 ms/query median, topic precision@5 0.928, source doc in top-10 150/150.
IVF: nlist=64, train 11.84s, build 0.187s.

| nprobe | recall@10 | topic P@5 | ms/query (median) | speedup | docs scanned |
|---:|---:|---:|---:|---:|---:|
| 1 | 0.8300 | 0.9840 | 0.616 | 3.6x | 1862 (1.9%) |
| 4 | 0.9650 | 0.9530 | 1.046 | 2.1x | 6503 (6.5%) |
| 8 | 0.9845 | 0.9390 | 1.464 | 1.5x | 12512 (12.5%) |
| 16 | 0.9945 | 0.9320 | 1.892 | 1.2x | 24930 (24.9%) |
| 32 | 0.9995 | 0.9280 | 2.839 | 0.8x | 49757 (49.8%) |
