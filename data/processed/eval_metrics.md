# Friends RAG — Retrieval Evaluation Results

Sample size: 800 questions (originals: 400, paraphrased: 400)

### All

| Mode | R@1 | R@3 | R@5 | R@10 | MRR |
|---|---|---|---|---|---|
| vector | 0.190 | 0.299 | 0.354 | 0.420 | 0.259 |
| vector_naive | 0.220 | 0.328 | 0.375 | 0.415 | 0.283 |
| vector_utterance_noheader | 0.220 | 0.286 | 0.326 | 0.380 | 0.265 |
| vector_window_noheader | 0.318 | 0.439 | 0.489 | 0.551 | 0.391 |
| bm25 | 0.297 | 0.414 | 0.461 | 0.530 | 0.367 |
| hybrid | 0.274 | 0.380 | 0.424 | 0.516 | 0.344 |
| hybrid_window_bm25 | 0.351 | 0.471 | 0.511 | 0.583 | 0.425 |
| hybrid_window_bm25_smart | 0.364 | 0.480 | 0.531 | 0.589 | 0.434 |
| hybrid_window_bm25_reranked | 0.378 | 0.486 | 0.535 | 0.595 | 0.443 |
| hybrid_window_bm25_reranked_bge | 0.366 | 0.499 | 0.552 | 0.608 | 0.444 |

### Originals

| Mode | R@1 | R@3 | R@5 | R@10 | MRR |
|---|---|---|---|---|---|
| vector | 0.172 | 0.275 | 0.323 | 0.390 | 0.239 |
| vector_naive | 0.205 | 0.312 | 0.375 | 0.412 | 0.272 |
| vector_utterance_noheader | 0.215 | 0.282 | 0.325 | 0.370 | 0.260 |
| vector_window_noheader | 0.315 | 0.432 | 0.482 | 0.552 | 0.387 |
| bm25 | 0.302 | 0.422 | 0.453 | 0.525 | 0.368 |
| hybrid | 0.263 | 0.375 | 0.410 | 0.502 | 0.333 |
| hybrid_window_bm25 | 0.340 | 0.468 | 0.515 | 0.580 | 0.419 |
| hybrid_window_bm25_smart | 0.350 | 0.477 | 0.535 | 0.593 | 0.426 |
| hybrid_window_bm25_reranked | 0.388 | 0.492 | 0.535 | 0.598 | 0.452 |
| hybrid_window_bm25_reranked_bge | 0.393 | 0.515 | 0.557 | 0.608 | 0.462 |

### Paraphrased

| Mode | R@1 | R@3 | R@5 | R@10 | MRR |
|---|---|---|---|---|---|
| vector | 0.207 | 0.323 | 0.385 | 0.450 | 0.280 |
| vector_naive | 0.235 | 0.343 | 0.375 | 0.417 | 0.293 |
| vector_utterance_noheader | 0.225 | 0.290 | 0.328 | 0.390 | 0.270 |
| vector_window_noheader | 0.320 | 0.445 | 0.495 | 0.550 | 0.396 |
| bm25 | 0.292 | 0.405 | 0.470 | 0.535 | 0.366 |
| hybrid | 0.285 | 0.385 | 0.438 | 0.530 | 0.354 |
| hybrid_window_bm25 | 0.362 | 0.475 | 0.507 | 0.585 | 0.431 |
| hybrid_window_bm25_smart | 0.378 | 0.482 | 0.527 | 0.585 | 0.443 |
| hybrid_window_bm25_reranked | 0.367 | 0.480 | 0.535 | 0.593 | 0.434 |
| hybrid_window_bm25_reranked_bge | 0.340 | 0.482 | 0.547 | 0.608 | 0.425 |