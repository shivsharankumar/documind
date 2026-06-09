"""
The full retrieval pipeline:
  query → hybrid search (dense + sparse) → rerank → top results
"""

from documind.embeddings import Embedder
from documind.reranker import Reranker
from documind.vectorstore import SearchResult, VectorStore


class Retriever:
    """
    One class that orchestrates the full retrieve-then-rerank pipeline.
    This is what RAG (Day 4) will call to get relevant chunks.
    """

    def __init__(self):
        self.embedder = Embedder()
        self.store = VectorStore()
        self.reranker = Reranker()

    async def setup(self) -> None:
        """Must be called once before using."""
        await self.store.setup()

    async def retrieve(
        self,
        query: str,
        k_candidates: int = 20,  # how many to fetch from vector DB
        k_final: int = 5,  # how many to return after reranking
        score_threshold: float = 0.2,
        source_filter: str | None = None,
    ) -> list[SearchResult]:
        """
        Full pipeline: embed query → hybrid search → rerank → return best.

        k_candidates >> k_final is intentional:
        fetch many cheap candidates, rerank to get the best few expensive ones.
        """
        # Step 1: embed the query (RETRIEVAL_QUERY mode)
        query_embedding = await self.embedder.embed(query, task_type="RETRIEVAL_QUERY")

        # Step 2: hybrid search → k_candidates results
        candidates = await self.store.search(
            query_embedding=query_embedding,
            query_text=query,
            k=k_candidates,
            source_filter=source_filter,
        )

        if not candidates:
            return []

        # Step 3: rerank → k_final results
        final = await self.reranker.rerank(
            query=query,
            results=candidates,
            top_n=k_final,
        )
        if final and final[0].score < score_threshold:
            return []
        return final
