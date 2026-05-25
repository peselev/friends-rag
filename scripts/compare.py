"""List all Chroma collections and their item counts."""
import chromadb
from src.config import CHROMA_DIR

client = chromadb.PersistentClient(path=str(CHROMA_DIR))
for c in client.list_collections():
    col = client.get_collection(c.name)
    print(f"{c.name}: {col.count():,} items")
