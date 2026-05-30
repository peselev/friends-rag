---
title: Friends RAG
emoji: ☕
colorFrom: indigo
colorTo: gray
sdk: docker
app_port: 8501
tags:
- streamlit
- rag
- retrieval-augmented-generation
pinned: false
short_description: Ask anything about Friends. Answers come with citations - and "I don't know" when transcripts don't say.
license: mit
---

# Friends — Grounded Q&A

Retrieval-augmented question answering over all 236 episodes of *Friends*.
Every answer is grounded in actual scene transcripts, with citations to the
exact episode and scene. If the transcripts don't contain the answer, the
system says so rather than guessing.

**Full write-up:** the design decisions, the evaluation, and what I learned building this are documented at [peselev.com/work/friends-rag](https://peselev.com/work/friends-rag/).

## How it works

- **Retrieval**: hybrid (BM25 keyword + window-level vector embeddings via OpenAI), with parent-scene resolution
- **Optional re-ranking**: cross-encoder (`BAAI/bge-reranker-base`) for higher accuracy
- **Generation**: Claude with a strict "use only retrieved scenes, cite them, refuse if insufficient" prompt

Built as a hands-on RAG learning project exploring retrieval design tradeoffs.
