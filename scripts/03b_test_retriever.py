"""
Test the full pipeline:
hybrid search + reranking working together.
"""

import asyncio
import sys
from pathlib import Path

sys.path.append(str(Path(__file__).resolve().parent.parent / "src"))
from dotenv import load_dotenv

from documind.embeddings import Embedder
from documind.retriever import Retriever
from documind.vectorstore import Chunk, VectorStore

load_dotenv()
DOCS = [
    "A vector database stores high-dimensional vectors and finds nearest neighbors fast.",
    "RAG stands for Retrieval-Augmented Generation. It feeds relevant docs into the LLM.",
    "Python asyncio is used for concurrent I/O-bound work without threads.",
    "FastAPI is a modern Python web framework built on Starlette and Pydantic.",
    "Embedding models turn text into vectors that capture semantic meaning.",
    "Pizza is a popular Italian dish made with dough, tomato sauce, and cheese.",
    "BM25 is a keyword-based ranking algorithm used in search engines for decades.",
    "Reranking improves search quality by scoring candidates with a cross-encoder model.",
    "Cosine similarity measures the angle between two vectors, used for semantic search.",
    "LLMs are stateless — they have no memory between separate API calls.",
]


async def main():
    # First: index some documents
    embedder = Embedder()
    store = VectorStore(dim=embedder.dim)
    await store.setup()

    chunks = [Chunk(text=d, source="test", chunk_index=i) for i, d in enumerate(DOCS)]
    embeddings = await embedder.embed_many([c.text for c in chunks])
    await store.add(chunks, embeddings)
    print(f"Indexed {len(chunks)} docs\n")

    # Now: use the retriever (hybrid + rerank)
    retriever = Retriever()
    retriever.store = store  # reuse the same store

    queries = [
        "how does vector search work?",
        "what is RAG?",
        "keyword search algorithm",
    ]

    for q in queries:
        print(f"Query: '{q}'")
        results = await retriever.retrieve(q, k_candidates=10, k_final=3)
        for i, r in enumerate(results, 1):
            print(f"  {i}. [rerank score: {r.score:.3f}] {r.chunk.text[:75]}...")
        print()


if __name__ == "__main__":
    asyncio.run(main())
