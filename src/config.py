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
SCENE_COLLECTION = "friends_scenes"
NAIVE_COLLECTION = "friends_naive"
