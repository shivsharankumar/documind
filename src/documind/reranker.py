"""Rerank search results using Cohere — find the truly relevant ones."""

import cohere

from documind.config import settings
from documind.vectorstore import SearchResult


class Reranker:
    """
    Takes vector search results and reranks them by true relevance.

    Think of it as: vector search = fast librarian grabbing candidates
                    reranker    = careful scholar picking the best ones
    """

    def __init__(self):
        self.client = cohere.AsyncClient(api_key=settings.cohere_api_key)
        self.model = "rerank-v3.5"  # Cohere's latest rerank model

    async def rerank(
        self,
        query: str,
        results: list[SearchResult],
        top_n: int = 5,
    ) -> list[SearchResult]:
        """
        Rerank search results by relevance to the query.

        Args:
            query: the user's original question
            results: the candidates from vector search (usually top 20)
            top_n: how many to return after reranking (usually 5)

        Returns:
            A smaller, better-ranked list of SearchResults
        """
        if not results:
            return []

        # Cohere takes plain strings — we extract just the text
        documents = [r.chunk.text for r in results]

        response = await self.client.rerank(
            model=self.model,
            query=query,
            documents=documents,
            top_n=top_n,
            return_documents=False,  # we already have the docs, saves bandwidth
        )

        # Cohere returns indices into our original list + new scores
        reranked = []
        for hit in response.results:
            original_result = results[hit.index]
            # Replace vector similarity score with reranker relevance score
            reranked.append(
                SearchResult(
                    chunk=original_result.chunk,
                    score=hit.relevance_score,  # 0 to 1, higher = more relevant
                )
            )

        return reranked
