
from pathlib import Path

import chromadb
from sentence_transformers import SentenceTransformer


# Use the same model for indexing and querying.
EMBEDDING_MODEL = (
    "sentence-transformers/all-MiniLM-L6-v2"
)

BASE_DIR = Path(__file__).resolve().parent

DB_PATH = BASE_DIR / "chroma_db"

COLLECTION_NAME = "payment_knowledge_v1"


def get_embedding_model():
    return SentenceTransformer(EMBEDDING_MODEL)


def get_collection():
    # PersistentClient stores the database on disk.
    client = chromadb.PersistentClient(
        path=str(DB_PATH)
    )

    # Reuse the collection if it already exists.
    # We supply embeddings ourselves rather than
    # using Chroma's default embedding function.
    # HNSW (Hierarchical Navigable Small World) is a graph-based algorithm for approximate nearest neighbor search.
    return client.get_or_create_collection(
        name=COLLECTION_NAME,
        configuration={
            "hnsw": {
                "space": "cosine" # how "close" is measured
            }
        },
    )
