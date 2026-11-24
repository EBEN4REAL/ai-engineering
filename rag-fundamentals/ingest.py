
import hashlib
from pathlib import Path

import pymupdf

from chunking import chunk_text

from vector_store import (
    BASE_DIR,
    get_collection,
    get_embedding_model,
)


DOCUMENTS_DIR = BASE_DIR / "documents"

CHUNK_SIZE = 500
CHUNK_OVERLAP = 100
EMBEDDING_BATCH_SIZE = 32


# -------------------------------------
# 1. Extract text from each PDF page
# -------------------------------------
def extract_pdf_pages(pdf_path: Path):
    pages = []

    # with opens a resource, lets you use it, and closes it automatically when the block ends, even if an error happens partway through.
    with pymupdf.open(pdf_path) as pdf:
        for page_number, page in enumerate(
            pdf, start=1
        ):
            text = page.get_text("text")

            if text.strip():
                pages.append({
                    "page": page_number,
                    "text": text,
                })

    return pages


# -------------------------------------
# 2. Split pages into smaller chunks
# -------------------------------------
def prepare_chunks(pdf_path: Path):
    records = []

    pages = extract_pdf_pages(pdf_path)

    print(
        f"Extracted {len(pages)} pages "
        f"from {pdf_path.name}"
    )

    for page in pages:
        chunks = chunk_text(
            page["text"],
            chunk_size=CHUNK_SIZE,
            overlap=CHUNK_OVERLAP,
        )

        print("CHUNKS => ", chunks)

        for index, content in enumerate(chunks):
            # Stable ID for this page/chunk content.
            identifier = (
                f"{pdf_path.name}:"
                f"{page['page']}:{index}:{content}"
            )

            chunk_id = hashlib.sha256(  # 2. hash those bytes
                identifier.encode("utf-8") # 1. text → bytes
            ).hexdigest() # # 3. result → readable hex string

            records.append({
                "id": chunk_id,
                "content": content,
                "metadata": {
                    "source": pdf_path.name,
                    "page": page["page"],
                    "chunk": index,
                },
            })

    return records


# -------------------------------------
# 3. Generate and store embeddings
# -------------------------------------
def ingest_documents():
    pdf_paths = sorted(DOCUMENTS_DIR.glob("*.pdf"))

    print("PDF paths:", pdf_paths)

    if not pdf_paths:
        print("No PDF files found in documents/.")
        return

    model = get_embedding_model()
    collection = get_collection()

    for pdf_path in pdf_paths:
        print(f"\nProcessing: {pdf_path.name}")

        records = prepare_chunks(pdf_path)

        if not records:
            print(
                "No extractable text found. "
                "Skipping this PDF."
            )
            continue

        # Encode this document's chunks before
        # replacing previously indexed records.
        contents = [
            record["content"]
            for record in records
        ]

        embeddings = model.encode_document(
            contents,
            batch_size=EMBEDDING_BATCH_SIZE,
            normalize_embeddings=True,
            convert_to_numpy=True,
        )

        # Remove old chunks for this source so
        # edited PDFs don't retain stale passages.
        collection.delete(
            where={"source": pdf_path.name}
        )

        # Store records in manageable batches.
        # batch size is the number of chunks to process at once, which can help with memory usage and speed.
        for start in range(
            0, len(records), EMBEDDING_BATCH_SIZE
        ):
            end = start + EMBEDDING_BATCH_SIZE

            batch = records[start:end]

            collection.upsert(
                ids=[
                    item["id"]
                    for item in batch
                ],
                documents=[
                    item["content"]
                    for item in batch
                ],
                embeddings=embeddings[
                    start:end
                ].tolist(),
                metadatas=[
                    item["metadata"]
                    for item in batch
                ],
            )

        print(
            f"Indexed {len(records)} chunks "
            f"from {pdf_path.name}"
        )

    print(
        "\nTotal indexed chunks:",
        collection.count()
    )


if __name__ == "__main__":
    ingest_documents()
