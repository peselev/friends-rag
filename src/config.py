"""
Central configuration. All settings and API keys live here so we only
read environment variables in one place.
"""
import logging
import os
from pathlib import Path
from dotenv import load_dotenv

# Silence Chroma telemetry warnings (version mismatch issue, not our concern)
logging.getLogger("chromadb.telemetry").setLevel(logging.CRITICAL)

# Load .env file from project root
PROJECT_ROOT = Path(__file__).parent.parent
load_dotenv(PROJECT_ROOT / ".env")

# API Keys
ANTHROPIC_API_KEY = os.getenv("ANTHROPIC_API_KEY")
OPENAI_API_KEY = os.getenv("OPENAI_API_KEY")

# Fail fast if keys are missing
if not ANTHROPIC_API_KEY:
    raise ValueError("ANTHROPIC_API_KEY not found in .env file")
if not OPENAI_API_KEY:
    raise ValueError("OPENAI_API_KEY not found in .env file")

# Paths
DATA_RAW = PROJECT_ROOT / "data" / "raw"
DATA_PROCESSED = PROJECT_ROOT / "data" / "processed"
CHUNKS_JSONL = DATA_PROCESSED / "chunks.jsonl"
BM25_INDEX_PATH = DATA_PROCESSED / "bm25_index.pkl"
CHROMA_DIR = PROJECT_ROOT / "chroma_db"

# Model choices
EMBEDDING_MODEL = "text-embedding-3-small"   # OpenAI: cheap, 1536 dimensions
GENERATION_MODEL = "claude-haiku-4-5"        # Anthropic: fast + cheap for demo

# Retrieval settings
TOP_K = 5   # Number of chunks to retrieve per query
SEARCH_EF = 400   # HNSW query-time search depth, baked into each collection at
                  # creation (Chroma 0.5.x can't change it afterward). Tuned via
                  # scripts.search_ef_sweep: recovers ~full exact-search recall
                  # for ~+3.5 ms/query, negligible next to rerank + generation.
SCENE_COLLECTION = "friends_scenes"
NAIVE_COLLECTION = "friends_naive"

# ⚠️ HEADER VARIANTS — DO NOT USE. These collections embed chunks with a
# "Friends S01E01, Scene 1" prefix. Measured WORSE than the no-header variants
# for window chunks (the boilerplate is a large fraction of a short chunk and
# pulls the embedding toward the header instead of the dialogue). The matching
# modes are commented out of AVAILABLE_MODES so they can't be run by accident.
# Kept only as a paper trail. Use the *_NOHEADER_COLLECTION constants below.
UTTERANCE_COLLECTION = "friends_utterance"   # ⚠️ header — do not use
WINDOW_COLLECTION = "friends_window"         # ⚠️ header — do not use

# Header-less variants — THESE ARE THE ONES TO USE.
UTTERANCE_NOHEADER_COLLECTION = "friends_utterance_noheader"
WINDOW_NOHEADER_COLLECTION = "friends_window_noheader"