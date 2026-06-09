"""
Side-by-side comparison of three retrieval approaches:
  A) Pure vector search (dense only, no reranking)
  B) Hybrid search (dense + BM25, no reranking)
  C) Hybrid search + Cohere reranking  ← what DocuMind uses

Run this to see exactly what each approach gets right and wrong.
"""

import asyncio
import sys
from pathlib import Path

# import contextlib

sys.path.append(str(Path(__file__).resolve().parent.parent / "src"))
import uuid

from dotenv import load_dotenv
from qdrant_client import AsyncQdrantClient
from qdrant_client.models import (
    Distance,
    Fusion,
    FusionQuery,
    PointStruct,
    Prefetch,
    SparseVectorParams,
    VectorParams,
)

from documind.embeddings import Embedder
from documind.reranker import Reranker
from documind.vectorstore import Chunk, SparseVector, _bm25_sparse_vector

load_dotenv()

# ── same docs as before ───────────────────────────────────────────────────────
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

QUERIES = [
    "how does vector search work?",
    "what is RAG?",
    "keyword search algorithm",
    "food",  # intentional off-topic test
    "memory in language models",  # needs semantic understanding
]

COLLECTION = "comparison_test"
TOP_K = 3  # results to show per approach


# ── helpers ───────────────────────────────────────────────────────────────────


def divider(title: str):
    print(f"\n{'═' * 60}")
    print(f"  {title}")
    print("═" * 60)


def show_results(label: str, results: list, score_label: str = "score"):
    print(f"\n  ┌─ {label}")
    for i, r in enumerate(results, 1):
        text = r.chunk.text if hasattr(r, "chunk") else r["text"]
        score = r.score if hasattr(r, "score") else r["score"]
        # truncate for display
        short = text[:65] + "..." if len(text) > 65 else text
        bar = "█" * int(score * 20)  # visual score bar
        print(f"  │  {i}. [{score:.3f}] {bar}")
        print(f"  │     {short}")
    print("  └" + "─" * 55)


# ── index once ────────────────────────────────────────────────────────────────


async def index_docs(embedder: Embedder) -> AsyncQdrantClient:
    """Create a fresh comparison collection with both dense + sparse."""
    client = AsyncQdrantClient(host="localhost", port=6333)

    # wipe if exists
    # try:
    #     await client.delete_collection(COLLECTION)
    # except Exception:
    #     pass
    # with contextlib.suppress(Exception):
    #     await client.delete_collection(COLLECTION)
    await client.create_collection(
        collection_name=COLLECTION,
        vectors_config={
            "dense": VectorParams(size=embedder.dim, distance=Distance.COSINE),
        },
        sparse_vectors_config={
            "sparse": SparseVectorParams(),
        },
    )

    chunks = [Chunk(text=d, source="compare", chunk_index=i) for i, d in enumerate(DOCS)]
    embeddings = await embedder.embed_many([c.text for c in chunks])

    points = [
        PointStruct(
            id=str(uuid.uuid4()),
            vector={
                "dense": emb,
                "sparse": _bm25_sparse_vector(chunk.text),
            },
            payload={"text": chunk.text, "source": chunk.source, "chunk_index": i},
        )
        for i, (chunk, emb) in enumerate(zip(chunks, embeddings, strict=False))
    ]
    await client.upsert(collection_name=COLLECTION, points=points)
    print(f"Indexed {len(chunks)} docs into '{COLLECTION}'\n")
    return client


# ── approach A: pure vector search ───────────────────────────────────────────


async def pure_vector_search(
    client: AsyncQdrantClient,
    query_embedding: list[float],
    k: int = TOP_K,
) -> list:
    response = await client.query_points(
        collection_name=COLLECTION,
        query=query_embedding,
        using="dense",  # ← dense only, no BM25
        limit=k,
    )
    return [
        type(
            "R",
            (),
            {
                "chunk": type("C", (), {"text": p.payload["text"]})(),
                "score": p.score,
            },
        )()
        for p in response.points
    ]


# ── approach B: hybrid search, no reranking ───────────────────────────────────


async def hybrid_no_rerank(
    client: AsyncQdrantClient,
    query_embedding: list[float],
    query_text: str,
    k: int = TOP_K,
) -> list:
    response = await client.query_points(
        collection_name=COLLECTION,
        prefetch=[
            Prefetch(query=query_embedding, using="dense", limit=k * 4),
            Prefetch(
                query=SparseVector(**_bm25_sparse_vector(query_text).__dict__),
                using="sparse",
                limit=k * 4,
            ),
        ],
        query=FusionQuery(fusion=Fusion.RRF),
        limit=k,
    )
    return [
        type(
            "R",
            (),
            {
                "chunk": type("C", (), {"text": p.payload["text"]})(),
                "score": p.score,
            },
        )()
        for p in response.points
    ]


# ── approach C: hybrid + rerank ───────────────────────────────────────────────


async def hybrid_with_rerank(
    client: AsyncQdrantClient,
    reranker: Reranker,
    query_embedding: list[float],
    query_text: str,
    k_candidates: int = 10,
    k_final: int = TOP_K,
) -> list:
    # fetch more candidates for the reranker to work with
    candidates = await hybrid_no_rerank(client, query_embedding, query_text, k=k_candidates)
    if not candidates:
        return []

    # rerank — convert to SearchResult-like objects
    from documind.vectorstore import Chunk as VChunk
    from documind.vectorstore import SearchResult

    sr_candidates = [
        SearchResult(
            chunk=VChunk(
                text=r.chunk.text,
                source="compare",
                chunk_index=0,
            ),
            score=r.score,
        )
        for r in candidates
    ]

    return await reranker.rerank(
        query=query_text,
        results=sr_candidates,
        top_n=k_final,
    )


# ── main comparison ───────────────────────────────────────────────────────────


async def main():
    embedder = Embedder()
    reranker = Reranker()
    client = await index_docs(embedder)

    for query in QUERIES:
        divider(f'Query: "{query}"')

        # embed query once, reuse for all three approaches
        q_emb = await embedder.embed(query, task_type="RETRIEVAL_QUERY")

        # A — pure vector
        vec_results = await pure_vector_search(client, q_emb)
        show_results("A) Pure Vector (semantic only)", vec_results)

        # B — hybrid, no rerank
        hyb_results = await hybrid_no_rerank(client, q_emb, query)
        show_results("B) Hybrid BM25 + Vector (no rerank)", hyb_results)

        # C — hybrid + rerank
        final_results = await hybrid_with_rerank(client, reranker, q_emb, query)
        show_results("C) Hybrid + Cohere Rerank ✓ (DocuMind)", final_results)

        # winner analysis
        print("\n  📊 Analysis:")
        top_a = vec_results[0].chunk.text[:50] if vec_results else "—"
        top_b = hyb_results[0].chunk.text[:50] if hyb_results else "—"
        top_c = final_results[0].chunk.text[:50] if final_results else "—"
        same_abc = top_a[:40] == top_b[:40] == top_c[:40]
        if same_abc:
            print("     All three agree on #1 — clear winner exists")
        else:
            if top_a[:40] != top_c[:40]:
                print(f"     Vector alone picked:  '{top_a}...'")
                print(f"     Reranker corrected to: '{top_c}...'")
            if top_b[:40] != top_c[:40]:
                print(f"     Hybrid alone picked:   '{top_b}...'")
                print(f"     Reranker corrected to:  '{top_c}...'")

    # clean up
    await client.delete_collection(COLLECTION)
    print(f"\n\nClean up: deleted '{COLLECTION}' collection")
    print("\nKey takeaways:")
    print("  • Pure vector: good at meaning, misses exact keywords")
    print("  • Hybrid: catches both meaning + keywords, but ranks by overlap not relevance")
    print("  • Hybrid + Rerank: understands actual relevance to YOUR question")


if __name__ == "__main__":
    asyncio.run(main())
