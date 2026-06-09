"""Ingest a PDF into DocuMind."""

import asyncio
import sys
from pathlib import Path

sys.path.append(str(Path(__file__).resolve().parent.parent / "src"))
from documind.embeddings import Embedder
from documind.ingest import ingest_pdf
from documind.vectorstore import VectorStore


async def main():
    if len(sys.argv) < 2:
        print("Usage: uv run python scripts/04_ingest.py <path_to_pdf>")
        return

    pdf_path = sys.argv[1]

    embedder = Embedder()
    store = VectorStore(dim=embedder.dim)
    await store.setup()

    print(f"Ingesting {pdf_path}...")
    n = await ingest_pdf(pdf_path, embedder, store)
    print(f"\nDone — {n} chunks now searchable.")


if __name__ == "__main__":
    asyncio.run(main())
