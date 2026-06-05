# Friends RAG - Reranker experiment

Candidate depth N=50, direct=400, reworded=400 (heavily_reworded excluded).

| Retrieval mode | Direct (recall@5) | Reworded (recall@5) | MRR (Direct) |
|---|---|---|---|
| Hybrid, no rerank | 0.530 | 0.507 | 0.434 |
| + MS-MARCO MiniLM | 0.542 | 0.537 | 0.452 |
| + BGE | 0.560 | 0.547 | 0.462 |
| + Cohere | 0.635 | 0.620 | 0.547 |
