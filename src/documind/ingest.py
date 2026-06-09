"""
Document ingestion: PDF → text → chunks → embeddings → Qdrant.
This runs once per document, ahead of query time.
"""

from pathlib import Path

from langchain_text_splitters import RecursiveCharacterTextSplitter
from pypdf import PdfReader

from documind.embeddings import Embedder
from documind.vectorstore import Chunk, VectorStore


def extract_pdf_text(pdf_path: str) -> list[tuple[int, str]]:
    """
    Extract text from a PDF, page by page.
    Returns a list of (page_number, text) so we can cite page numbers later.
    """
    reader = PdfReader(pdf_path)
    pages = []
    for i, page in enumerate(reader.pages, start=1):
        text = page.extract_text() or ""
        if text.strip():  # skip blank pages
            pages.append((i, text))
    return pages


def chunk_pages(
    pages: list[tuple[int, str]],
    source_name: str,
    chunk_size: int = 300,
    chunk_overlap: int = 120,
) -> list[Chunk]:
    """
    Split page text into overlapping chunks using recursive splitting.
    Keeps track of which page each chunk came from (for citations).
    """
    splitter = RecursiveCharacterTextSplitter(
        chunk_size=chunk_size,
        chunk_overlap=chunk_overlap,
        # Try these separators in order — paragraph, line, sentence, word
        separators=["\n\n", "\n", ". ", " ", ""],
    )

    chunks: list[Chunk] = []
    chunk_idx = 0
    for page_num, page_text in pages:
        pieces = splitter.split_text(page_text)
        for piece in pieces:
            chunks.append(
                Chunk(
                    text=piece,
                    source=source_name,
                    chunk_index=chunk_idx,
                    extra={"page": page_num},  # ← page number for citation
                )
            )
            chunk_idx += 1
    return chunks


async def ingest_pdf(
    pdf_path: str,
    embedder: Embedder,
    store: VectorStore,
    chunk_size: int = 300,
    chunk_overlap: int = 120,
) -> int:
    """
    Full ingestion pipeline for one PDF.
    Returns the number of chunks stored.
    """
    source_name = Path(pdf_path).name

    # 1. Extract
    pages = extract_pdf_text(pdf_path)
    print(f"  Extracted {len(pages)} pages from {source_name}")

    # 2. Chunk
    chunks = chunk_pages(pages, source_name, chunk_size, chunk_overlap)
    print(f"  Split into {len(chunks)} chunks")

    # 3. Embed (in batches — embedding 1000 chunks at once may hit limits)
    BATCH = 100
    all_embeddings = []
    for i in range(0, len(chunks), BATCH):
        batch_texts = [c.text for c in chunks[i : i + BATCH]]
        batch_emb = await embedder.embed_many(batch_texts)
        all_embeddings.extend(batch_emb)
        print(f"  Embedded {min(i + BATCH, len(chunks))}/{len(chunks)} chunks")

    # 4. Store
    await store.add(chunks, all_embeddings)
    print(f"  Stored {len(chunks)} chunks in Qdrant")

    return len(chunks)
