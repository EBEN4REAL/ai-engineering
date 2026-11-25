"""
Hybrid search with reranking.

For every query:

    query
      ├─► Retriever 1: semantic search (Chroma)  → top 10 chunks by MEANING
      └─► Retriever 2: keyword search (BM25)     → top 10 chunks by EXACT WORDS
                         │
                         ▼
          Reciprocal Rank Fusion (RRF)           → one merged list (up to 20)
                         │
                         ▼
          Cross-encoder reranker                 → best 3, scored precisely

Run ingest.py first so Chroma contains chunks.
"""

import re

import numpy as np
from rank_bm25 import BM25Okapi
from sentence_transformers import CrossEncoder

from search_db import search_documents
from vector_store import get_collection


# -----------------------------------
# Configuration
# -----------------------------------
RERANKER_MODEL = "cross-encoder/ms-marco-MiniLM-L6-v2"

CANDIDATE_K = 10  # results fetched from EACH retriever
FINAL_TOP_K = 3   # results kept after reranking
RRF_K = 60        # RRF smoothing constant (60 is the standard value)


# -----------------------------------
# 1. Load indexed chunks from Chroma
# -----------------------------------
def load_documents() -> list[dict]:
    # BM25 needs the text of every chunk in memory,
    # so read everything ingest.py stored.
    # Fine for a small project; large systems use a
    # dedicated keyword index (e.g. Elasticsearch).
    collection = get_collection()

    result = collection.get(
        include=["documents", "metadatas"]
    )

    documents = []

    # Chroma returns parallel lists (columns);
    # zip turns them back into one dict per chunk.
    for chunk_id, content, metadata in zip(
        result["ids"],
        result["documents"],
        result["metadatas"],
    ):
        documents.append({
            "id": chunk_id,
            "content": content,
            "source": metadata["source"],
            "page": metadata["page"],
            "chunk": metadata["chunk"],
        })

    return documents


# -----------------------------------
# 2. Retriever 2: keyword search (BM25)
# -----------------------------------
def tokenize_for_bm25(text: str) -> list[str]:
    # Lowercase, then extract runs of letters/digits.
    # Hyphenated codes such as DD-1042 stay one token.
    return re.findall(
        r"[a-z0-9]+(?:-[a-z0-9]+)*",
        text.lower()
    )


class BM25Search:
    def __init__(self, documents: list[dict]):
        if not documents:
            raise ValueError(
                "Cannot build BM25 with no documents."
            )

        self.documents = documents

        # Convert every chunk into a list of tokens.
        tokenized_corpus = [
            tokenize_for_bm25(doc["content"])
            for doc in documents
        ]

        # Build the in-memory keyword index ONCE.
        # It pre-computes word statistics for the
        # whole corpus (how rare each word is, etc.).
        self.bm25 = BM25Okapi(tokenized_corpus)

    def search(
        self,
        query: str,
        top_k: int = CANDIDATE_K,
    ) -> list[dict]:
        tokens = tokenize_for_bm25(query)

        if not tokens:
            return []

        # One BM25 score per chunk, in corpus order.
        # E.g. [0.2, 4.5, 0.0, 1.3, 0.8]
        scores = self.bm25.get_scores(tokens)

        print("BM25 scores:", scores)

        # Chunk positions sorted from highest to lowest score.
        ranked_indices = np.argsort(scores)[::-1]

        results = []

        for index in ranked_indices:
            index = int(index)
            score = float(scores[index])

            # Skip chunks that share no words with the query.
            if score <= 0:
                continue

            results.append({
                **self.documents[index],
                "bm25_score": score,
            })

            if len(results) >= top_k:
                break

        return results


# -----------------------------------
# 3. Fusion: Reciprocal Rank Fusion (RRF)
# -----------------------------------
def reciprocal_rank_fusion(
    ranked_lists: list[list[dict]],
    k: int = RRF_K,
) -> list[dict]:
    scores = {}     # chunk ID → accumulated RRF score
    documents = {}  # chunk ID → merged chunk fields

    for ranked_list in ranked_lists:
        for rank, document in enumerate(
            ranked_list, start=1
        ):
            doc_id = document["id"]

            # Each list adds 1 / (k + rank).
            # Rank 1 → 1/61, rank 2 → 1/62, ...
            # A chunk found by both lists gets both shares.
            scores[doc_id] = (
                scores.get(doc_id, 0.0)
                + 1.0 / (k + rank)
            )

            # Merge rather than overwrite, so a chunk found
            # by both retrievers keeps "distance" AND "bm25_score".
            documents[doc_id] = {
                **documents.get(doc_id, {}),
                **document,
            }

    combined = [
        {**documents[doc_id], "rrf_score": score}
        for doc_id, score in scores.items()
    ]

    return sorted(
        combined,
        key=lambda doc: doc["rrf_score"],
        reverse=True,
    )


# -----------------------------------
# 4. Hybrid search: both retrievers + fusion
# -----------------------------------
def hybrid_search(
    query: str,
    bm25_engine: BM25Search,
    candidate_k: int = CANDIDATE_K,
) -> list[dict]:

    # Retriever 1: semantic search (meaning), from search_db.py.
    semantic_results = search_documents(
        query,
        top_k=candidate_k,
    )

    # Retriever 2: keyword search (exact words).
    keyword_results = bm25_engine.search(
        query,
        top_k=candidate_k,
    )

    # Merge both rankings into one list, matched by chunk ID.
    return reciprocal_rank_fusion([
        semantic_results,
        keyword_results,
    ])


# -----------------------------------
# 5. Reranking with a cross-encoder
# -----------------------------------
class PassageReranker:
    def __init__(self, model_name: str = RERANKER_MODEL):
        # A cross-encoder reads the query and a passage
        # TOGETHER through the transformer and outputs one
        # relevance score. More accurate than comparing
        # separate embeddings, but much slower, so it only
        # scores the small candidate list.
        self.model = CrossEncoder(model_name)

    def rerank(
        self,
        query: str,
        candidates: list[dict],
        top_k: int = FINAL_TOP_K,
    ) -> list[dict]:

        if not candidates:
            return []

        # Pair the query with every candidate passage.
        pairs = [
            (query, doc["content"])
            for doc in candidates
        ]

        # One score per pair. Higher = more relevant.
        scores = self.model.predict(pairs)

        ranked = [
            {**document, "rerank_score": float(score)}
            for document, score in zip(candidates, scores)
        ]

        ranked.sort(
            key=lambda doc: doc["rerank_score"],
            reverse=True,
        )

        return ranked[:top_k]


# -----------------------------------
# 6. Display results
# -----------------------------------
def print_results(results: list[dict]) -> None:
    for rank, result in enumerate(results, start=1):
        print(f"\n{rank}. {result['source']} (page {result['page']})")

        # Show which retriever(s) found this chunk.
        found_by = []

        if "distance" in result:
            found_by.append(
                f"semantic (distance {result['distance']:.4f})"
            )

        if "bm25_score" in result:
            found_by.append(
                f"keyword (BM25 {result['bm25_score']:.2f})"
            )

        print("Found by:", ", ".join(found_by))
        print("RRF score:", f"{result['rrf_score']:.4f}")
        print("Rerank score:", f"{result['rerank_score']:.4f}")
        print("Content:", result["content"])


# -----------------------------------
# 7. Interactive application
# -----------------------------------
def main() -> None:
    documents = load_documents()

    if not documents:
        raise SystemExit(
            "No documents found. Run ingest.py first."
        )

    # Build both expensive components ONCE, before the loop.
    bm25_engine = BM25Search(documents)
    reranker = PassageReranker()

    print("Advanced Hybrid Search")
    print("Indexed chunks:", len(documents))
    print("Type 'exit' to quit.")

    while True:
        query = input("\nSearch: ").strip()

        if query.lower() in {"exit", "quit"}:
            break

        if not query:
            continue

        # Stage 1: cast a wide net (recall).
        candidates = hybrid_search(query, bm25_engine)

        # Stage 2: score precisely, keep the best (precision).
        results = reranker.rerank(query, candidates)

        print_results(results)


if __name__ == "__main__":
    main()
