
def chunk_text(
    text: str,
    chunk_size: int = 500,
    overlap: int = 100,
) -> list[str]:
    # Validate the chunking configuration.
    if chunk_size <= 0:
        raise ValueError("chunk_size must be positive.")

    if not 0 <= overlap < chunk_size:
        raise ValueError(
            "overlap must be smaller than chunk_size."
        )

    # Normalize whitespace to make splitting predictable.
    text = " ".join(text.split())

    print("Normalized text length:", text, len(text))

    if not text:
        return []

    chunks = []

    step = chunk_size - overlap

    # Walk through the text using overlapping windows.
    for start in range(0, len(text), step):
        end = min(start + chunk_size, len(text))

        print("step:", step)

        print(f"Chunking text from {start} to {end} (length {end - start})")

        chunk = text[start:end]
        
        chunks.append(chunk)

        if end == len(text):
            break

    return chunks


def chunk_by_words(
    text: str,
    chunk_size: int = 100,
    overlap: int = 20,
) -> list[str]:
    """
    Divide text into overlapping word-based chunks.

    chunk_size: Maximum number of words per chunk.
    overlap: Number of words shared by adjacent chunks.
    """
    if chunk_size <= 0:
        raise ValueError("chunk_size must be positive")

    if not 0 <= overlap < chunk_size:
        raise ValueError(
            "overlap must be smaller than chunk_size"
        )

    words = text.split()

    if not words:
        return []

    chunks = []
    step = chunk_size - overlap

    for start in range(0, len(words), step):
        end = min(start + chunk_size, len(words))

        chunk = " ".join(words[start:end])
        chunks.append(chunk)

        if end == len(words):
            break

    return chunks


if __name__ == "__main__":
    text = (
        "A direct debit may fail because the customer's "
        "account has insufficient funds. The system "
        "records the failure and may notify the customer."
    )

    print("Original text:", len(text), "characters")

    chunks = chunk_text(
        text,
        chunk_size=60,
        overlap=15,
    )

    for index, chunk in enumerate(chunks, start=1):
        print(f"\nChunk {index}:")
        print(chunk)
