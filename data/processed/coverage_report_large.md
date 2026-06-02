# Friends RAG - Union Coverage Experiment

Total questions: 1000  |  all=1000, direct=400, reworded=400, heavily_reworded=200

Unit: scene-level. `k` counts unique parent scenes per retriever (chunks deduped to scenes). For window/utterance this reads slightly higher than the chunk-rank numbers in the writeup; for bm25 and scene-vector it is identical.

Pool: BM25 (keyword), Vector (scene), Vector (window), Vector (utterance)

Query embedding model: `text-embedding-3-large`  |  vector collections suffix: `_large`



## ALL  (n=1000)

### Per-mode recall@k (scene-level)

| Mode | @1 | @5 | @10 | @20 | @50 |
|---|---|---|---|---|---|
| BM25 (keyword) | 0.238 | 0.373 | 0.428 | 0.475 | 0.533 |
| Vector (scene) | 0.185 | 0.318 | 0.386 | 0.438 | 0.520 |
| Vector (window) | 0.332 | 0.482 | 0.547 | 0.593 | 0.672 |
| Vector (utterance) | 0.236 | 0.368 | 0.417 | 0.474 | 0.550 |

### Union coverage@k  (ceiling a reranker could reach)

| Pool | @1 | @5 | @10 | @20 | @50 |
|---|---|---|---|---|---|
| bm25 only (baseline) | 0.238 | 0.373 | 0.428 | 0.475 | 0.533 |
| bm25 + window (shipped pair) | 0.401 | 0.544 | 0.614 | 0.662 | 0.737 |
| bm25 + window + scene | 0.432 | 0.579 | 0.644 | 0.696 | 0.765 |
| FULL pool (all 4) | 0.456 | 0.606 | 0.669 | 0.713 | 0.786 |

### Consensus @10: how many of the 4 retrievers find gold

| # retrievers hitting | questions | share |
|---|---|---|
| 0 | 331 | 33.1% |
| 1 | 135 | 13.5% |
| 2 | 155 | 15.5% |
| 3 | 183 | 18.3% |
| 4 | 196 | 19.6% |

### Consensus @20: how many of the 4 retrievers find gold

| # retrievers hitting | questions | share |
|---|---|---|
| 0 | 287 | 28.7% |
| 1 | 128 | 12.8% |
| 2 | 150 | 15.0% |
| 3 | 188 | 18.8% |
| 4 | 247 | 24.7% |

### Marginal contribution @10: questions only this mode finds

| Mode | unique-hit questions |
|---|---|
| BM25 (keyword) | 39 |
| Vector (scene) | 27 |
| Vector (window) | 44 |
| Vector (utterance) | 25 |

### Redundancy @10 (mode pairs)

gold co-hit = of questions where either finds gold, share where BOTH do (high = redundant on correctness). set overlap = mean Jaccard of the top-10 retrieved scene sets (high = return the same scenes).

| Pair | gold co-hit | set overlap |
|---|---|---|
| BM25 (keyword) + Vector (scene) | 0.477 | 0.094 |
| BM25 (keyword) + Vector (window) | 0.588 | 0.114 |
| BM25 (keyword) + Vector (utterance) | 0.512 | 0.090 |
| Vector (scene) + Vector (window) | 0.584 | 0.182 |
| Vector (scene) + Vector (utterance) | 0.465 | 0.108 |
| Vector (window) + Vector (utterance) | 0.626 | 0.248 |


## DIRECT  (n=400)

### Per-mode recall@k (scene-level)

| Mode | @1 | @5 | @10 | @20 | @50 |
|---|---|---|---|---|---|
| BM25 (keyword) | 0.302 | 0.453 | 0.525 | 0.590 | 0.660 |
| Vector (scene) | 0.203 | 0.345 | 0.405 | 0.455 | 0.560 |
| Vector (window) | 0.360 | 0.550 | 0.613 | 0.660 | 0.738 |
| Vector (utterance) | 0.268 | 0.405 | 0.453 | 0.520 | 0.590 |

### Union coverage@k  (ceiling a reranker could reach)

| Pool | @1 | @5 | @10 | @20 | @50 |
|---|---|---|---|---|---|
| bm25 only (baseline) | 0.302 | 0.453 | 0.525 | 0.590 | 0.660 |
| bm25 + window (shipped pair) | 0.455 | 0.632 | 0.698 | 0.748 | 0.818 |
| bm25 + window + scene | 0.490 | 0.665 | 0.723 | 0.782 | 0.845 |
| FULL pool (all 4) | 0.522 | 0.685 | 0.738 | 0.790 | 0.855 |

### Consensus @10: how many of the 4 retrievers find gold

| # retrievers hitting | questions | share |
|---|---|---|
| 0 | 105 | 26.2% |
| 1 | 47 | 11.8% |
| 2 | 82 | 20.5% |
| 3 | 77 | 19.2% |
| 4 | 89 | 22.2% |

### Consensus @20: how many of the 4 retrievers find gold

| # retrievers hitting | questions | share |
|---|---|---|
| 0 | 84 | 21.0% |
| 1 | 47 | 11.8% |
| 2 | 76 | 19.0% |
| 3 | 81 | 20.2% |
| 4 | 112 | 28.0% |

### Marginal contribution @10: questions only this mode finds

| Mode | unique-hit questions |
|---|---|
| BM25 (keyword) | 17 |
| Vector (scene) | 9 |
| Vector (window) | 15 |
| Vector (utterance) | 6 |

### Redundancy @10 (mode pairs)

gold co-hit = of questions where either finds gold, share where BOTH do (high = redundant on correctness). set overlap = mean Jaccard of the top-10 retrieved scene sets (high = return the same scenes).

| Pair | gold co-hit | set overlap |
|---|---|---|
| BM25 (keyword) + Vector (scene) | 0.453 | 0.095 |
| BM25 (keyword) + Vector (window) | 0.631 | 0.122 |
| BM25 (keyword) + Vector (utterance) | 0.552 | 0.093 |
| Vector (scene) + Vector (window) | 0.571 | 0.174 |
| Vector (scene) + Vector (utterance) | 0.453 | 0.099 |
| Vector (window) + Vector (utterance) | 0.608 | 0.231 |


## REWORDED  (n=400)

### Per-mode recall@k (scene-level)

| Mode | @1 | @5 | @10 | @20 | @50 |
|---|---|---|---|---|---|
| BM25 (keyword) | 0.292 | 0.470 | 0.535 | 0.580 | 0.642 |
| Vector (scene) | 0.230 | 0.372 | 0.453 | 0.515 | 0.570 |
| Vector (window) | 0.393 | 0.537 | 0.605 | 0.647 | 0.708 |
| Vector (utterance) | 0.263 | 0.405 | 0.468 | 0.515 | 0.560 |

### Union coverage@k  (ceiling a reranker could reach)

| Pool | @1 | @5 | @10 | @20 | @50 |
|---|---|---|---|---|---|
| bm25 only (baseline) | 0.292 | 0.470 | 0.535 | 0.580 | 0.642 |
| bm25 + window (shipped pair) | 0.470 | 0.608 | 0.685 | 0.725 | 0.777 |
| bm25 + window + scene | 0.502 | 0.632 | 0.708 | 0.743 | 0.795 |
| FULL pool (all 4) | 0.515 | 0.660 | 0.733 | 0.760 | 0.812 |

### Consensus @10: how many of the 4 retrievers find gold

| # retrievers hitting | questions | share |
|---|---|---|
| 0 | 107 | 26.8% |
| 1 | 57 | 14.2% |
| 2 | 45 | 11.2% |
| 3 | 87 | 21.8% |
| 4 | 104 | 26.0% |

### Consensus @20: how many of the 4 retrievers find gold

| # retrievers hitting | questions | share |
|---|---|---|
| 0 | 96 | 24.0% |
| 1 | 51 | 12.8% |
| 2 | 38 | 9.5% |
| 3 | 84 | 21.0% |
| 4 | 131 | 32.8% |

### Marginal contribution @10: questions only this mode finds

| Mode | unique-hit questions |
|---|---|
| BM25 (keyword) | 21 |
| Vector (scene) | 9 |
| Vector (window) | 17 |
| Vector (utterance) | 10 |

### Redundancy @10 (mode pairs)

gold co-hit = of questions where either finds gold, share where BOTH do (high = redundant on correctness). set overlap = mean Jaccard of the top-10 retrieved scene sets (high = return the same scenes).

| Pair | gold co-hit | set overlap |
|---|---|---|
| BM25 (keyword) + Vector (scene) | 0.574 | 0.122 |
| BM25 (keyword) + Vector (window) | 0.664 | 0.146 |
| BM25 (keyword) + Vector (utterance) | 0.560 | 0.114 |
| Vector (scene) + Vector (window) | 0.633 | 0.193 |
| Vector (scene) + Vector (utterance) | 0.508 | 0.119 |
| Vector (window) + Vector (utterance) | 0.669 | 0.261 |


## HEAVILY_REWORDED  (n=200)

### Per-mode recall@k (scene-level)

| Mode | @1 | @5 | @10 | @20 | @50 |
|---|---|---|---|---|---|
| BM25 (keyword) | 0.000 | 0.020 | 0.020 | 0.035 | 0.060 |
| Vector (scene) | 0.060 | 0.155 | 0.215 | 0.250 | 0.340 |
| Vector (window) | 0.155 | 0.235 | 0.300 | 0.350 | 0.470 |
| Vector (utterance) | 0.120 | 0.220 | 0.245 | 0.300 | 0.450 |

### Union coverage@k  (ceiling a reranker could reach)

| Pool | @1 | @5 | @10 | @20 | @50 |
|---|---|---|---|---|---|
| bm25 only (baseline) | 0.000 | 0.020 | 0.020 | 0.035 | 0.060 |
| bm25 + window (shipped pair) | 0.155 | 0.240 | 0.305 | 0.365 | 0.495 |
| bm25 + window + scene | 0.175 | 0.300 | 0.360 | 0.430 | 0.545 |
| FULL pool (all 4) | 0.205 | 0.340 | 0.405 | 0.465 | 0.595 |

### Consensus @10: how many of the 4 retrievers find gold

| # retrievers hitting | questions | share |
|---|---|---|
| 0 | 119 | 59.5% |
| 1 | 31 | 15.5% |
| 2 | 28 | 14.0% |
| 3 | 19 | 9.5% |
| 4 | 3 | 1.5% |

### Consensus @20: how many of the 4 retrievers find gold

| # retrievers hitting | questions | share |
|---|---|---|
| 0 | 107 | 53.5% |
| 1 | 30 | 15.0% |
| 2 | 36 | 18.0% |
| 3 | 23 | 11.5% |
| 4 | 4 | 2.0% |

### Marginal contribution @10: questions only this mode finds

| Mode | unique-hit questions |
|---|---|
| BM25 (keyword) | 1 |
| Vector (scene) | 9 |
| Vector (window) | 12 |
| Vector (utterance) | 9 |

### Redundancy @10 (mode pairs)

gold co-hit = of questions where either finds gold, share where BOTH do (high = redundant on correctness). set overlap = mean Jaccard of the top-10 retrieved scene sets (high = return the same scenes).

| Pair | gold co-hit | set overlap |
|---|---|---|
| BM25 (keyword) + Vector (scene) | 0.068 | 0.035 |
| BM25 (keyword) + Vector (window) | 0.049 | 0.035 |
| BM25 (keyword) + Vector (utterance) | 0.060 | 0.032 |
| Vector (scene) + Vector (window) | 0.451 | 0.178 |
| Vector (scene) + Vector (utterance) | 0.353 | 0.106 |
| Vector (window) + Vector (utterance) | 0.535 | 0.257 |