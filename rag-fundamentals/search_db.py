
from vector_store import (
    get_collection,
    get_embedding_model,
)


# Load the model and existing database.
model = get_embedding_model()
collection = get_collection()


def search_documents(question: str, top_k: int = 3):
    if not question.strip():
        raise ValueError("Question cannot be empty.")

    if top_k <= 0:
        raise ValueError("top_k must be positive.")

    count = collection.count()

    if count == 0:
        return []

    # 1. Embed the user's search query.
    query_embedding = model.encode_query(
        question,
        normalize_embeddings=True,
        convert_to_numpy=True,
    )

    print("Query embedding:", query_embedding)

    # 2. Search previously stored embeddings.
    results = collection.query(
        query_embeddings=[
            query_embedding.tolist()
        ],
        n_results=min(top_k, count),
        include=[
            "documents",
            "metadatas",
            "distances",
        ],
    )

    print("Raw search results:", results)

    # 3. Convert Chroma's nested query output
    # into a simpler list of search results.
    matches = []

    for chunk_id, content, metadata, distance in zip(
        results["ids"][0],
        results["documents"][0],
        results["metadatas"][0],
        results["distances"][0],
    ):
        matches.append({
            "id": chunk_id,
            "content": content,
            "source": metadata["source"],
            "page": metadata["page"],
            "chunk": metadata["chunk"],
            "distance": float(distance),
        })

    return matches


if __name__ == "__main__":
    print("Persistent PDF Semantic Search")
    print("Indexed chunks:", collection.count())
    print("Type 'exit' to quit.")

    while True:
        question = input("\nSearch: ").strip()

        if question.lower() in {"exit", "quit"}:
            break

        if not question:
            continue

        matches = search_documents(question)

        if not matches:
            print("No indexed documents found.")
            continue

        for rank, match in enumerate(
            matches, start=1
        ):
            print(f"\n{rank}. {match['source']}")
            print("Page:", match["page"])
            print("Chunk:", match["chunk"])
            print(
                "Cosine distance:",
                f"{match['distance']:.4f}" # :.4f means "show this number with exactly 4 digits after the decimal point."
            )
            print("Content:", match["content"])
