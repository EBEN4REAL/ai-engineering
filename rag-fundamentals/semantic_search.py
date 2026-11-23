
import os
# os.environ["HF_HUB_OFFLINE"] = "1"    # must come before importing sentence_transformers

import numpy as np
from sentence_transformers import SentenceTransformer

# A real system often sets a minimum score so it can say "no good match" instead of returning something irrelevant.

# ----------------------------------
# 1. Simulated knowledge base
# ----------------------------------
# Each document contains an ID, a title,
# and the information we want to search.

#  In a real RAG system these would be chunks of your help-centre articles, PDFs, and so on
documents = [
    {
        "id": 1,
        "title": "Failed Direct Debit Transactions",
        "content": (
            "A direct debit payment may fail because "
            "the customer's account has insufficient "
            "funds, the mandate is inactive, or "
            "the issuing bank rejects the debit."
        )
    },
    {
        "id": 2,
        "title": "Cancel a Payment Mandate",
        "content": (
            "Customers can cancel their active "
            "direct debit mandates through an "
            "authorized cancellation process."
        )
    },
    {
        "id": 3,
        "title": "Payment Refunds",
        "content": (
            "A customer may request a refund "
            "when an eligible payment was "
            "processed incorrectly."
        )
    },
    {
        "id": 4,
        "title": "Payment Authentication",
        "content": (
            "Some payment transactions require "
            "customer authentication using "
            "a one-time password or another "
            "verification method."
        )
    },
    {
        "id": 5,
        "title": "Payment Gateway Troubleshooting",
        "content": (
            "Payment gateway failures can occur "
            "because of network timeouts, "
            "service unavailability, or "
            "upstream processing errors."
        )
    },
    {
        "id": 6,
        "title": "Insufficient Account Balance",
        "content": (
            "A transaction can be rejected when "
            "the customer's available account balance "
            "is lower than the payment amount."
        )
    }
]


# ----------------------------------
# 2. Load pretrained embedding model
# ----------------------------------
MODEL_NAME = "sentence-transformers/all-MiniLM-L6-v2" 
# Hugging Face model that converts text into 384-dimensional vectors. It is trained to produce similar vectors for semantically similar sentences.
# source => https://huggingface.co/sentence-transformers/all-MiniLM-L6-v2/blob/main/README.md?utm_source=chatgpt.com

model = SentenceTransformer(MODEL_NAME)


# ----------------------------------
# 3. Convert documents into embeddings
# ----------------------------------
# Encode the title and content together,
# allowing both to influence retrieval.

document_texts = [
    f"{doc['title']}. {doc['content']}"
    for doc in documents
]

# Each document becomes a vector of 384 numbers.
# normalize_embeddings=True scales each vector
# to have approximately unit length.

document_embeddings = model.encode_document(
    document_texts,
    normalize_embeddings=True, # scales that vector to length 1
    convert_to_numpy=True
)

print("Document embeddings shape:", document_embeddings.shape)


# ----------------------------------
# 4. Semantic search function
# ----------------------------------
def semantic_search(query: str, top_k: int = 3):

    if not query.strip():
        raise ValueError("Search query cannot be empty.")

    # Convert the user's query into an embedding.
    query_embedding = model.encode_query(
        query,
        normalize_embeddings=True,
        convert_to_numpy=True
    )

    print("Query embedding shape:", query_embedding.shape)

    # Calculate similarities using matrix multiplication.
    # Because all vectors are normalized,
    # the dot product equals cosine similarity.
    similarities = document_embeddings @ query_embedding

    print("Similarities:", similarities)

    # Sort document indices by similarity.
    # Higher similarity means closer vector direction.
    ranked_indices = np.argsort(similarities)[::-1] # doesn't return the sorted scores. It returns the positions(indices) of the scores in order from highest to lowest.

    print("Ranked indices:", ranked_indices)

    # Return only the top matching documents.
    results = []

    for index in ranked_indices[:top_k]:
        document = documents[int(index)] # int(index) converts NumPy's int64 into a plain Python int

        results.append({
            "id": document["id"],
            "title": document["title"],
            "content": document["content"],
            "score": float(similarities[index])
        })

    return results


# ----------------------------------
# 5. Interactive search
# ----------------------------------
if __name__ == "__main__":

    print("\nSemantic Search Engine")
    print("Type 'exit' to quit.\n")

    while True:
        query = input("Search: ").strip()

        if query.lower() in {"exit", "quit"}:
            break

        if not query:
            continue

        results = semantic_search(query, top_k=3)

        print("\nTop matching documents:")

        for rank, result in enumerate(results, start=1):
            print(f"\n{rank}. {result['title']}")
            print(f"Similarity: {result['score']:.4f}")
            print(f"Content: {result['content']}")

        print("\n" + "-" * 50)
