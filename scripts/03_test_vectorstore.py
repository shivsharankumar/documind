"""Smoke test: embed a few sentences, store them, search."""

import asyncio
import sys
from pathlib import Path

sys.path.append(str(Path(__file__).resolve().parent.parent / "src"))
from dotenv import load_dotenv

from documind.embeddings import Embedder
from documind.vectorstore import Chunk, VectorStore

load_dotenv()  # load Gemini API key from .env
DOCS = [
    "A vector database stores high-dimensional vectors and finds nearest neighbors fast.",
    "RAG stands for Retrieval-Augmented Generation. It feeds relevant docs into the LLM prompt.",
    "Python's asyncio module is used for concurrent I/O-bound work.",
    "FastAPI is a modern Python web framework built on Starlette and Pydantic.",
    "Embedding models turn text into vectors that capture semantic meaning.",
    "Pizza is a popular Italian dish made with dough, tomato sauce, and cheese.",
]


# /Users/shivsharankumar/Documents/documind/src/documind/vectorstore.py
async def main():
    embedder = Embedder()
    store = VectorStore(dim=embedder.dim)
    await store.setup()

    # Build chunks
    chunks = [Chunk(text=d, source="smoke_test", chunk_index=i) for i, d in enumerate(DOCS)]

    # Embed them all in one batch (fast + cheap)
    embeddings = await embedder.embed_many([c.text for c in chunks])

    # Store them
    await store.add(chunks, embeddings)
    print(f"Stored {len(chunks)} chunks.\n")

    # Now query
    queries = [
        "what is a vector DB?",
        "how does retrieval augmented generation work?",
        "food",
    ]

    for q in queries:
        q_emb = await embedder.embed(q)
        results = await store.search(q_emb, k=3)
        print(f"Query: {q}")
        for r in results:
            print(f"  [{r.score:.3f}] {r.chunk.text[:80]}...")
        print()


if __name__ == "__main__":
    asyncio.run(main())
