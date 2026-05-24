# Friends RAG

A Retrieval-Augmented Generation system over the transcripts of the TV show *Friends*, built to demonstrate and compare different retrieval strategies (vector search, keyword search, hybrid) with rigorous evaluation against a Q&A benchmark.

## Setup

1. Clone this repo
2. Copy `.env.example` to `.env` and add your API keys
3. `pip install -r requirements.txt`
4. `python -m src.loader` to download and process the dataset
5. `python -m src.embedder` to build the vector index
6. `streamlit run app.py` to launch the UI

## Data

Transcripts from [Emory NLP character-mining](https://github.com/emorynlp/character-mining):
236 episodes, 3,107 scenes, ~1.1M tokens.

## Status

Work in progress.
