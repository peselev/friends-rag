"""
Disk-backed cache for QUERY embeddings.

Why: during evaluation we embed the same ~1,000 questions on every re-run, and
every vector mode embeds the identical query text with the same model. Embedding
is the slow, paid step. Caching by (model, text) means each unique question is
sent to OpenAI at most once, ever — re-runs become free and instant.

Scope: query-time embeddings ONLY. Chunk/document embeddings are built once by
the indexer and already persist inside Chroma, so they are deliberately not
cached here.
"""
import atexit
import hashlib
import pickle
import threading
from pathlib import Path

from src.config import DATA_PROCESSED, EMBEDDING_MODEL

CACHE_PATH = DATA_PROCESSED / "query_embed_cache.pkl"


class EmbeddingCache:
    def __init__(self, path: Path = CACHE_PATH):
        self.path = path
        self._store: dict[str, list[float]] = {}
        self._dirty = False
        self._lock = threading.Lock()
        if path.exists():
            try:
                with open(path, "rb") as f:
                    self._store = pickle.load(f)
            except Exception:
                self._store = {}  # corrupt/old cache: start clean, never crash

    @staticmethod
    def _key(text: str, model: str) -> str:
        h = hashlib.sha256()
        h.update(model.encode("utf-8"))
        h.update(b"\x00")               # separator so model+text can't collide
        h.update(text.encode("utf-8"))
        return h.hexdigest()

    def get(self, text: str, model: str = EMBEDDING_MODEL) -> list[float] | None:
        return self._store.get(self._key(text, model))

    def set(self, text: str, vector: list[float], model: str = EMBEDDING_MODEL) -> None:
        with self._lock:
            self._store[self._key(text, model)] = vector
            self._dirty = True

    def save(self) -> None:
        with self._lock:
            if not self._dirty:
                return
            tmp = self.path.with_suffix(self.path.suffix + ".tmp")
            with open(tmp, "wb") as f:
                pickle.dump(self._store, f)
            tmp.replace(self.path)       # atomic on POSIX
            self._dirty = False

    def __len__(self) -> int:
        return len(self._store)


# Process-wide singleton, shared by every retriever. Flushed to disk on exit so
# no caller has to remember to save().
_cache = EmbeddingCache()
atexit.register(_cache.save)