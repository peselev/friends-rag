# Friends RAG - Union Coverage Experiment

Total questions: 1000  |  all=1000, direct=400, reworded=400, heavily_reworded=200

Unit: scene-level. `k` counts unique parent scenes per retriever (chunks deduped to scenes). For window/utterance this reads slightly higher than the chunk-rank numbers in the writeup; for bm25 and scene-vector it is identical.

Pool: BM25 (keyword), Vector (scene), Vector (window), Vector (utterance)



## ALL  (n=1000)

### Per-mode recall@k (scene-level)

| Mode | @1 | @5 | @10 | @20 | @50 |
|---|---|---|---|---|---|
| BM25 (keyword) | 0.238 | 0.373 | 0.428 | 0.475 | 0.533 |
| Vector (scene) | 0.148 | 0.277 | 0.323 | 0.366 | 0.425 |
| Vector (window) | 0.278 | 0.433 | 0.497 | 0.563 | 0.646 |
| Vector (utterance) | 0.192 | 0.293 | 0.350 | 0.398 | 0.478 |

### Union coverage@k  (ceiling a reranker could reach)

| Pool | @1 | @5 | @10 | @20 | @50 |
|---|---|---|---|---|---|
| bm25 only (baseline) | 0.238 | 0.373 | 0.428 | 0.475 | 0.533 |
| bm25 + window (shipped pair) | 0.367 | 0.515 | 0.576 | 0.642 | 0.715 |
| bm25 + window + scene | 0.400 | 0.552 | 0.613 | 0.671 | 0.741 |
| FULL pool (all 4) | 0.430 | 0.579 | 0.636 | 0.691 | 0.760 |

### Consensus @10: how many of the 4 retrievers find gold

| # retrievers hitting | questions | share |
|---|---|---|
| 0 | 364 | 36.4% |
| 1 | 150 | 15.0% |
| 2 | 153 | 15.3% |
| 3 | 190 | 19.0% |
| 4 | 143 | 14.3% |

### Consensus @20: how many of the 4 retrievers find gold

| # retrievers hitting | questions | share |
|---|---|---|
| 0 | 309 | 30.9% |
| 1 | 138 | 13.8% |
| 2 | 177 | 17.7% |
| 3 | 194 | 19.4% |
| 4 | 182 | 18.2% |

### Marginal contribution @10: questions only this mode finds

| Mode | unique-hit questions |
|---|---|
| BM25 (keyword) | 51 |
| Vector (scene) | 31 |
| Vector (window) | 45 |
| Vector (utterance) | 23 |

### Redundancy @10 (mode pairs)

gold co-hit = of questions where either finds gold, share where BOTH do (high = redundant on correctness). set overlap = mean Jaccard of the top-10 retrieved scene sets (high = return the same scenes).

| Pair | gold co-hit | set overlap |
|---|---|---|
| BM25 (keyword) + Vector (scene) | 0.428 | 0.076 |
| BM25 (keyword) + Vector (window) | 0.606 | 0.109 |
| BM25 (keyword) + Vector (utterance) | 0.451 | 0.077 |
| Vector (scene) + Vector (window) | 0.494 | 0.150 |
| Vector (scene) + Vector (utterance) | 0.393 | 0.082 |
| Vector (window) + Vector (utterance) | 0.560 | 0.214 |


## DIRECT  (n=400)

### Per-mode recall@k (scene-level)

| Mode | @1 | @5 | @10 | @20 | @50 |
|---|---|---|---|---|---|
| BM25 (keyword) | 0.302 | 0.453 | 0.525 | 0.590 | 0.660 |
| Vector (scene) | 0.158 | 0.300 | 0.355 | 0.393 | 0.470 |
| Vector (window) | 0.320 | 0.490 | 0.560 | 0.630 | 0.718 |
| Vector (utterance) | 0.217 | 0.323 | 0.378 | 0.432 | 0.520 |

### Union coverage@k  (ceiling a reranker could reach)

| Pool | @1 | @5 | @10 | @20 | @50 |
|---|---|---|---|---|---|
| bm25 only (baseline) | 0.302 | 0.453 | 0.525 | 0.590 | 0.660 |
| bm25 + window (shipped pair) | 0.438 | 0.600 | 0.655 | 0.725 | 0.800 |
| bm25 + window + scene | 0.470 | 0.625 | 0.680 | 0.745 | 0.825 |
| FULL pool (all 4) | 0.497 | 0.637 | 0.693 | 0.752 | 0.835 |

### Consensus @10: how many of the 4 retrievers find gold

| # retrievers hitting | questions | share |
|---|---|---|
| 0 | 123 | 30.8% |
| 1 | 51 | 12.8% |
| 2 | 69 | 17.2% |
| 3 | 90 | 22.5% |
| 4 | 67 | 16.8% |

### Consensus @20: how many of the 4 retrievers find gold

| # retrievers hitting | questions | share |
|---|---|---|
| 0 | 99 | 24.8% |
| 1 | 48 | 12.0% |
| 2 | 74 | 18.5% |
| 3 | 94 | 23.5% |
| 4 | 85 | 21.2% |

### Marginal contribution @10: questions only this mode finds

| Mode | unique-hit questions |
|---|---|
| BM25 (keyword) | 26 |
| Vector (scene) | 8 |
| Vector (window) | 12 |
| Vector (utterance) | 5 |

### Redundancy @10 (mode pairs)

gold co-hit = of questions where either finds gold, share where BOTH do (high = redundant on correctness). set overlap = mean Jaccard of the top-10 retrieved scene sets (high = return the same scenes).

| Pair | gold co-hit | set overlap |
|---|---|---|
| BM25 (keyword) + Vector (scene) | 0.443 | 0.082 |
| BM25 (keyword) + Vector (window) | 0.656 | 0.120 |
| BM25 (keyword) + Vector (utterance) | 0.462 | 0.083 |
| Vector (scene) + Vector (window) | 0.525 | 0.146 |
| Vector (scene) + Vector (utterance) | 0.409 | 0.076 |
| Vector (window) + Vector (utterance) | 0.569 | 0.213 |


## REWORDED  (n=400)

### Per-mode recall@k (scene-level)

| Mode | @1 | @5 | @10 | @20 | @50 |
|---|---|---|---|---|---|
| BM25 (keyword) | 0.292 | 0.470 | 0.535 | 0.580 | 0.642 |
| Vector (scene) | 0.193 | 0.333 | 0.375 | 0.415 | 0.445 |
| Vector (window) | 0.323 | 0.500 | 0.557 | 0.618 | 0.682 |
| Vector (utterance) | 0.233 | 0.335 | 0.398 | 0.443 | 0.495 |

### Union coverage@k  (ceiling a reranker could reach)

| Pool | @1 | @5 | @10 | @20 | @50 |
|---|---|---|---|---|---|
| bm25 only (baseline) | 0.292 | 0.470 | 0.535 | 0.580 | 0.642 |
| bm25 + window (shipped pair) | 0.427 | 0.593 | 0.657 | 0.715 | 0.765 |
| bm25 + window + scene | 0.468 | 0.625 | 0.693 | 0.738 | 0.777 |
| FULL pool (all 4) | 0.507 | 0.660 | 0.713 | 0.757 | 0.795 |

### Consensus @10: how many of the 4 retrievers find gold

| # retrievers hitting | questions | share |
|---|---|---|
| 0 | 115 | 28.7% |
| 1 | 61 | 15.2% |
| 2 | 61 | 15.2% |
| 3 | 89 | 22.2% |
| 4 | 74 | 18.5% |

### Consensus @20: how many of the 4 retrievers find gold

| # retrievers hitting | questions | share |
|---|---|---|
| 0 | 97 | 24.2% |
| 1 | 55 | 13.8% |
| 2 | 70 | 17.5% |
| 3 | 85 | 21.2% |
| 4 | 93 | 23.2% |

### Marginal contribution @10: questions only this mode finds

| Mode | unique-hit questions |
|---|---|
| BM25 (keyword) | 24 |
| Vector (scene) | 13 |
| Vector (window) | 16 |
| Vector (utterance) | 8 |

### Redundancy @10 (mode pairs)

gold co-hit = of questions where either finds gold, share where BOTH do (high = redundant on correctness). set overlap = mean Jaccard of the top-10 retrieved scene sets (high = return the same scenes).

| Pair | gold co-hit | set overlap |
|---|---|---|
| BM25 (keyword) + Vector (scene) | 0.462 | 0.092 |
| BM25 (keyword) + Vector (window) | 0.662 | 0.133 |
| BM25 (keyword) + Vector (utterance) | 0.510 | 0.095 |
| Vector (scene) + Vector (window) | 0.516 | 0.157 |
| Vector (scene) + Vector (utterance) | 0.405 | 0.089 |
| Vector (window) + Vector (utterance) | 0.585 | 0.220 |


## HEAVILY_REWORDED  (n=200)

### Per-mode recall@k (scene-level)

| Mode | @1 | @5 | @10 | @20 | @50 |
|---|---|---|---|---|---|
| BM25 (keyword) | 0.000 | 0.020 | 0.020 | 0.035 | 0.060 |
| Vector (scene) | 0.040 | 0.120 | 0.155 | 0.215 | 0.295 |
| Vector (window) | 0.105 | 0.185 | 0.250 | 0.320 | 0.430 |
| Vector (utterance) | 0.060 | 0.150 | 0.200 | 0.240 | 0.360 |

### Union coverage@k  (ceiling a reranker could reach)

| Pool | @1 | @5 | @10 | @20 | @50 |
|---|---|---|---|---|---|
| bm25 only (baseline) | 0.000 | 0.020 | 0.020 | 0.035 | 0.060 |
| bm25 + window (shipped pair) | 0.105 | 0.190 | 0.255 | 0.330 | 0.445 |
| bm25 + window + scene | 0.125 | 0.260 | 0.320 | 0.390 | 0.500 |
| FULL pool (all 4) | 0.140 | 0.300 | 0.370 | 0.435 | 0.540 |

### Consensus @10: how many of the 4 retrievers find gold

| # retrievers hitting | questions | share |
|---|---|---|
| 0 | 126 | 63.0% |
| 1 | 38 | 19.0% |
| 2 | 23 | 11.5% |
| 3 | 11 | 5.5% |
| 4 | 2 | 1.0% |

### Consensus @20: how many of the 4 retrievers find gold

| # retrievers hitting | questions | share |
|---|---|---|
| 0 | 113 | 56.5% |
| 1 | 35 | 17.5% |
| 2 | 33 | 16.5% |
| 3 | 15 | 7.5% |
| 4 | 4 | 2.0% |

### Marginal contribution @10: questions only this mode finds

| Mode | unique-hit questions |
|---|---|
| BM25 (keyword) | 1 |
| Vector (scene) | 10 |
| Vector (window) | 17 |
| Vector (utterance) | 10 |

### Redundancy @10 (mode pairs)

gold co-hit = of questions where either finds gold, share where BOTH do (high = redundant on correctness). set overlap = mean Jaccard of the top-10 retrieved scene sets (high = return the same scenes).

| Pair | gold co-hit | set overlap |
|---|---|---|
| BM25 (keyword) + Vector (scene) | 0.061 | 0.032 |
| BM25 (keyword) + Vector (window) | 0.059 | 0.037 |
| BM25 (keyword) + Vector (utterance) | 0.048 | 0.028 |
| Vector (scene) + Vector (window) | 0.286 | 0.146 |
| Vector (scene) + Vector (utterance) | 0.291 | 0.081 |
| Vector (window) + Vector (utterance) | 0.429 | 0.203 |