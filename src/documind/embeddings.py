"""Turn text into embedding vectors using Gemini."""

from google import genai
from google.genai import types

from documind.config import settings


class Embedder:
    """Wraps the Gemini embedding API. Reuse one instance everywhere."""

    def __init__(self):
        self.client = genai.Client(api_key=settings.gemini_api_key)
        self.model = "gemini-embedding-001"
        self.dim = 3072  # gemini-embedding-001 default output dimension

    async def embed(self, text: str, task_type: str = "RETRIEVAL_QUERY") -> list[float]:
        """Embed a single piece of text (used for search queries)."""
        result = await self.client.aio.models.embed_content(
            model=self.model,
            contents=text,
            config=types.EmbedContentConfig(task_type=task_type),
        )
        return result.embeddings[0].values

    async def embed_many(
        self, texts: list[str], task_type: str = "RETRIEVAL_DOCUMENT"
    ) -> list[list[float]]:
        """Embed several texts in one call (used for indexing documents)."""
        result = await self.client.aio.models.embed_content(
            model=self.model,
            contents=texts,
            config=types.EmbedContentConfig(task_type=task_type),
        )
        return [e.values for e in result.embeddings]
