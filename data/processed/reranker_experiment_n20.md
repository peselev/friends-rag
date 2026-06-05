# Friends RAG - Reranker experiment

Candidate depth N=20, direct=400, reworded=400 (heavily_reworded excluded).

| Retrieval mode | Direct (recall@5) | Reworded (recall@5) | MRR (Direct) |
|---|---|---|---|
| Hybrid, no rerank | 0.530 | 0.507 | 0.431 |
| + MS-MARCO MiniLM | 0.537 | 0.545 | 0.455 |
| + BGE | 0.552 | 0.568 | 0.473 |
| + Cohere | 0.620 | 0.625 | 0.533 |
