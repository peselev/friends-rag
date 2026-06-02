# Friends RAG - Union Coverage Experiment

Total questions: 1000  |  all=1000, direct=400, reworded=400, heavily_reworded=200

Unit: scene-level. `k` counts unique parent scenes per retriever (chunks deduped to scenes). For window/utterance this reads slightly higher than the chunk-rank numbers in the writeup; for bm25 and scene-vector it is identical.

Pool: BM25 (keyword), Vector (scene), Vector (window), Vector (utterance)

Query embedding model: `text-embedding-3-small`



## ALL  (n=1000)

### Per-mode recall@k (scene-level)

| Mode | @1 | @5 | @10 | @20 | @50 |
|---|---|---|---|---|---|
| BM25 (keyword) | 0.238 | 0.373 | 0.428 | 0.475 | 0.533 |
| Vector (scene) | 0.164 | 0.308 | 0.374 | 0.424 | 0.493 |
| Vector (window) | 0.281 | 0.436 | 0.499 | 0.565 | 0.648 |
| Vector (utterance) | 0.191 | 0.293 | 0.349 | 0.403 | 0.478 |

### Union coverage@k  (ceiling a reranker could reach)

| Pool | @1 | @5 | @10 | @20 | @50 |
|---|---|---|---|---|---|
| bm25 only (baseline) | 0.238 | 0.373 | 0.428 | 0.475 | 0.533 |
| bm25 + window (shipped pair) | 0.367 | 0.516 | 0.576 | 0.642 | 0.715 |
| bm25 + window + scene | 0.404 | 0.548 | 0.615 | 0.673 | 0.742 |
| FULL pool (all 4) | 0.433 | 0.578 | 0.637 | 0.693 | 0.762 |

### Consensus @10: how many of the 4 retrievers find gold

| # retrievers hitting | questions | share |
|---|---|---|
| 0 | 363 | 36.3% |
| 1 | 147 | 14.7% |
| 2 | 139 | 13.9% |
| 3 | 179 | 17.9% |
| 4 | 172 | 17.2% |

### Consensus @20: how many of the 4 retrievers find gold

| # retrievers hitting | questions | share |
|---|---|---|
| 0 | 307 | 30.7% |
| 1 | 136 | 13.6% |
| 2 | 155 | 15.5% |
| 3 | 187 | 18.7% |
| 4 | 215 | 21.5% |

### Marginal contribution @10: questions only this mode finds

| Mode | unique-hit questions |
|---|---|
| BM25 (keyword) | 50 |
| Vector (scene) | 31 |
| Vector (window) | 44 |
| Vector (utterance) | 22 |

### Redundancy @10 (mode pairs)

gold co-hit = of questions where either finds gold, share where BOTH do (high = redundant on correctness). set overlap = mean Jaccard of the top-10 retrieved scene sets (high = return the same scenes).

| Pair | gold co-hit | set overlap |
|---|---|---|
| BM25 (keyword) + Vector (scene) | 0.496 | 0.086 |
| BM25 (keyword) + Vector (window) | 0.609 | 0.109 |
| BM25 (keyword) + Vector (utterance) | 0.450 | 0.077 |
| Vector (scene) + Vector (window) | 0.576 | 0.164 |
| Vector (scene) + Vector (utterance) | 0.458 | 0.091 |
| Vector (window) + Vector (utterance) | 0.559 | 0.215 |


## DIRECT  (n=400)

### Per-mode recall@k (scene-level)

| Mode | @1 | @5 | @10 | @20 | @50 |
|---|---|---|---|---|---|
| BM25 (keyword) | 0.302 | 0.453 | 0.525 | 0.590 | 0.660 |
| Vector (scene) | 0.172 | 0.323 | 0.390 | 0.438 | 0.545 |
| Vector (window) | 0.323 | 0.492 | 0.562 | 0.632 | 0.720 |
| Vector (utterance) | 0.217 | 0.328 | 0.380 | 0.438 | 0.517 |

### Union coverage@k  (ceiling a reranker could reach)

| Pool | @1 | @5 | @10 | @20 | @50 |
|---|---|---|---|---|---|
| bm25 only (baseline) | 0.302 | 0.453 | 0.525 | 0.590 | 0.660 |
| bm25 + window (shipped pair) | 0.438 | 0.600 | 0.655 | 0.725 | 0.800 |
| bm25 + window + scene | 0.470 | 0.623 | 0.682 | 0.748 | 0.830 |
| FULL pool (all 4) | 0.500 | 0.640 | 0.695 | 0.752 | 0.840 |

### Consensus @10: how many of the 4 retrievers find gold

| # retrievers hitting | questions | share |
|---|---|---|
| 0 | 122 | 30.5% |
| 1 | 52 | 13.0% |
| 2 | 64 | 16.0% |
| 3 | 85 | 21.2% |
| 4 | 77 | 19.2% |

### Consensus @20: how many of the 4 retrievers find gold

| # retrievers hitting | questions | share |
|---|---|---|
| 0 | 99 | 24.8% |
| 1 | 48 | 12.0% |
| 2 | 66 | 16.5% |
| 3 | 89 | 22.2% |
| 4 | 98 | 24.5% |

### Marginal contribution @10: questions only this mode finds

| Mode | unique-hit questions |
|---|---|
| BM25 (keyword) | 26 |
| Vector (scene) | 9 |
| Vector (window) | 12 |
| Vector (utterance) | 5 |

### Redundancy @10 (mode pairs)

gold co-hit = of questions where either finds gold, share where BOTH do (high = redundant on correctness). set overlap = mean Jaccard of the top-10 retrieved scene sets (high = return the same scenes).

| Pair | gold co-hit | set overlap |
|---|---|---|
| BM25 (keyword) + Vector (scene) | 0.494 | 0.089 |
| BM25 (keyword) + Vector (window) | 0.660 | 0.120 |
| BM25 (keyword) + Vector (utterance) | 0.454 | 0.082 |
| Vector (scene) + Vector (window) | 0.574 | 0.158 |
| Vector (scene) + Vector (utterance) | 0.460 | 0.086 |
| Vector (window) + Vector (utterance) | 0.577 | 0.217 |


## REWORDED  (n=400)

### Per-mode recall@k (scene-level)

| Mode | @1 | @5 | @10 | @20 | @50 |
|---|---|---|---|---|---|
| BM25 (keyword) | 0.292 | 0.470 | 0.535 | 0.580 | 0.642 |
| Vector (scene) | 0.207 | 0.385 | 0.450 | 0.500 | 0.525 |
| Vector (window) | 0.328 | 0.502 | 0.560 | 0.620 | 0.685 |
| Vector (utterance) | 0.230 | 0.333 | 0.400 | 0.445 | 0.500 |

### Union coverage@k  (ceiling a reranker could reach)

| Pool | @1 | @5 | @10 | @20 | @50 |
|---|---|---|---|---|---|
| bm25 only (baseline) | 0.292 | 0.470 | 0.535 | 0.580 | 0.642 |
| bm25 + window (shipped pair) | 0.427 | 0.593 | 0.657 | 0.715 | 0.765 |
| bm25 + window + scene | 0.470 | 0.625 | 0.690 | 0.738 | 0.775 |
| FULL pool (all 4) | 0.507 | 0.660 | 0.713 | 0.760 | 0.795 |

### Consensus @10: how many of the 4 retrievers find gold

| # retrievers hitting | questions | share |
|---|---|---|
| 0 | 115 | 28.7% |
| 1 | 60 | 15.0% |
| 2 | 50 | 12.5% |
| 3 | 82 | 20.5% |
| 4 | 93 | 23.2% |

### Consensus @20: how many of the 4 retrievers find gold

| # retrievers hitting | questions | share |
|---|---|---|
| 0 | 96 | 24.0% |
| 1 | 55 | 13.8% |
| 2 | 57 | 14.2% |
| 3 | 79 | 19.8% |
| 4 | 113 | 28.2% |

### Marginal contribution @10: questions only this mode finds

| Mode | unique-hit questions |
|---|---|
| BM25 (keyword) | 23 |
| Vector (scene) | 12 |
| Vector (window) | 16 |
| Vector (utterance) | 9 |

### Redundancy @10 (mode pairs)

gold co-hit = of questions where either finds gold, share where BOTH do (high = redundant on correctness). set overlap = mean Jaccard of the top-10 retrieved scene sets (high = return the same scenes).

| Pair | gold co-hit | set overlap |
|---|---|---|
| BM25 (keyword) + Vector (scene) | 0.570 | 0.106 |
| BM25 (keyword) + Vector (window) | 0.665 | 0.134 |
| BM25 (keyword) + Vector (utterance) | 0.508 | 0.096 |
| Vector (scene) + Vector (window) | 0.636 | 0.173 |
| Vector (scene) + Vector (utterance) | 0.485 | 0.099 |
| Vector (window) + Vector (utterance) | 0.587 | 0.220 |


## HEAVILY_REWORDED  (n=200)

### Per-mode recall@k (scene-level)

| Mode | @1 | @5 | @10 | @20 | @50 |
|---|---|---|---|---|---|
| BM25 (keyword) | 0.000 | 0.020 | 0.020 | 0.035 | 0.060 |
| Vector (scene) | 0.060 | 0.125 | 0.190 | 0.245 | 0.325 |
| Vector (window) | 0.105 | 0.190 | 0.250 | 0.320 | 0.430 |
| Vector (utterance) | 0.060 | 0.145 | 0.185 | 0.250 | 0.355 |

### Union coverage@k  (ceiling a reranker could reach)

| Pool | @1 | @5 | @10 | @20 | @50 |
|---|---|---|---|---|---|
| bm25 only (baseline) | 0.000 | 0.020 | 0.020 | 0.035 | 0.060 |
| bm25 + window (shipped pair) | 0.105 | 0.195 | 0.255 | 0.330 | 0.445 |
| bm25 + window + scene | 0.140 | 0.245 | 0.330 | 0.395 | 0.500 |
| FULL pool (all 4) | 0.150 | 0.290 | 0.370 | 0.440 | 0.540 |

### Consensus @10: how many of the 4 retrievers find gold

| # retrievers hitting | questions | share |
|---|---|---|
| 0 | 126 | 63.0% |
| 1 | 35 | 17.5% |
| 2 | 25 | 12.5% |
| 3 | 12 | 6.0% |
| 4 | 2 | 1.0% |

### Consensus @20: how many of the 4 retrievers find gold

| # retrievers hitting | questions | share |
|---|---|---|
| 0 | 112 | 56.0% |
| 1 | 33 | 16.5% |
| 2 | 32 | 16.0% |
| 3 | 19 | 9.5% |
| 4 | 4 | 2.0% |

### Marginal contribution @10: questions only this mode finds

| Mode | unique-hit questions |
|---|---|
| BM25 (keyword) | 1 |
| Vector (scene) | 10 |
| Vector (window) | 16 |
| Vector (utterance) | 8 |

### Redundancy @10 (mode pairs)

gold co-hit = of questions where either finds gold, share where BOTH do (high = redundant on correctness). set overlap = mean Jaccard of the top-10 retrieved scene sets (high = return the same scenes).

| Pair | gold co-hit | set overlap |
|---|---|---|
| BM25 (keyword) + Vector (scene) | 0.050 | 0.037 |
| BM25 (keyword) + Vector (window) | 0.059 | 0.038 |
| BM25 (keyword) + Vector (utterance) | 0.051 | 0.028 |
| Vector (scene) + Vector (window) | 0.354 | 0.160 |
| Vector (scene) + Vector (utterance) | 0.339 | 0.083 |
| Vector (window) + Vector (utterance) | 0.381 | 0.200 |