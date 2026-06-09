"""Store and search embeddings using Qdrant — with hybrid search support."""

import uuid
from dataclasses import dataclass

from qdrant_client import AsyncQdrantClient
from qdrant_client.models import (
    Distance,
    FieldCondition,
    Filter,
    Fusion,
    FusionQuery,
    MatchValue,
    PointStruct,
    Prefetch,
    SparseVector,
    SparseVectorParams,
    VectorParams,
)


@dataclass
class Chunk:
    text: str
    source: str
    chunk_index: int
    extra: dict | None = None


@dataclass
class SearchResult:
    chunk: Chunk
    score: float


def _bm25_sparse_vector(text: str) -> SparseVector:
    """
    A simple BM25-style sparse vector from text.

    Real talk: production systems use a proper BM25 library here
    (like rank_bm25 or a dedicated tokenizer). For learning, this
    simple version shows the concept — each unique word becomes a
    dimension, count becomes the value. Qdrant normalizes it.
    """
    words = text.lower().split()
    word_counts: dict[int, float] = {}
    for word in words:
        # Convert word to a number (hash) — sparse vector needs int indices
        idx = abs(hash(word)) % 30000  # 30000-dim sparse space
        word_counts[idx] = word_counts.get(idx, 0) + 1.0

    return SparseVector(
        indices=list(word_counts.keys()),
        values=list(word_counts.values()),
    )


class VectorStore:
    """
    Qdrant wrapper with hybrid search (dense vector + sparse BM25).
    """

    DENSE_VEC = "dense"  # name for our embedding vectors
    SPARSE_VEC = "sparse"  # name for our BM25 sparse vectors

    def __init__(self, collection_name: str = "documind", dim: int = 3072):
        self.client = AsyncQdrantClient(host="localhost", port=6333)
        self.collection = collection_name
        self.dim = dim

    async def setup(self) -> None:
        """Create collection with both dense and sparse vector support."""
        collections = await self.client.get_collections()
        names = [c.name for c in collections.collections]
        if self.collection not in names:
            await self.client.create_collection(
                collection_name=self.collection,
                vectors_config={
                    # Dense vectors: our Gemini embeddings
                    self.DENSE_VEC: VectorParams(
                        size=self.dim,
                        distance=Distance.COSINE,
                    ),
                },
                sparse_vectors_config={
                    # Sparse vectors: BM25-style keyword matching
                    self.SPARSE_VEC: SparseVectorParams(),
                },
            )

    async def add(
        self,
        chunks: list[Chunk],
        embeddings: list[list[float]],
    ) -> None:
        """Insert chunks with both dense embeddings and sparse BM25 vectors."""
        if len(chunks) != len(embeddings):
            raise ValueError(
                f"chunks ({len(chunks)}) and embeddings ({len(embeddings)}) must match"
            )

        points = [
            PointStruct(
                id=str(uuid.uuid4()),
                vector={
                    # Dense: the Gemini embedding
                    self.DENSE_VEC: emb,
                    # Sparse: BM25-style from the raw text
                    self.SPARSE_VEC: _bm25_sparse_vector(chunk.text),
                },
                payload={
                    "text": chunk.text,
                    "source": chunk.source,
                    "chunk_index": chunk.chunk_index,
                    **(chunk.extra or {}),
                },
            )
            for chunk, emb in zip(chunks, embeddings, strict=False)
        ]

        await self.client.upsert(
            collection_name=self.collection,
            points=points,
        )

    async def search(
        self,
        query_embedding: list[float],
        query_text: str,  # needed for sparse/BM25 matching
        k: int = 20,  # fetch more candidates for reranker
        source_filter: str | None = None,
    ) -> list[SearchResult]:
        """
        Hybrid search: dense (semantic) + sparse (keyword) combined.
        Returns up to k candidates for the reranker to narrow down.
        """
        filt = None
        if source_filter:
            filt = Filter(
                must=[FieldCondition(key="source", match=MatchValue(value=source_filter))]
            )

        # Qdrant's Reciprocal Rank Fusion (RRF) merges results from
        # both dense and sparse search automatically.
        # Think of RRF as: "if a result ranks high in BOTH searches,
        # it definitely wins. If high in only one, it still gets a chance."
        response = await self.client.query_points(
            collection_name=self.collection,
            prefetch=[
                # Prefetch from dense (semantic) search
                Prefetch(
                    query=query_embedding,
                    using=self.DENSE_VEC,
                    limit=k,
                ),
                # Prefetch from sparse (keyword) search
                Prefetch(
                    query=SparseVector(**_bm25_sparse_vector(query_text).__dict__),
                    using=self.SPARSE_VEC,
                    limit=k,
                ),
            ],
            # RRF merges the two prefetch result lists
            query=FusionQuery(fusion=Fusion.RRF),
            limit=k,
            query_filter=filt,
        )

        return [
            SearchResult(
                chunk=Chunk(
                    text=r.payload["text"],
                    source=r.payload["source"],
                    chunk_index=r.payload["chunk_index"],
                ),
                score=r.score,
            )
            for r in response.points
        ]
