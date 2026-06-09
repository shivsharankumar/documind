"""
The RAG answer pipeline: retrieve → build prompt → generate cited answer.
This is where DocuMind actually answers questions.
"""

from dataclasses import dataclass

from groq import AsyncGroq

from documind.config import settings
from documind.retriever import Retriever
from documind.vectorstore import SearchResult


@dataclass
class RAGAnswer:
    answer: str
    sources: list[dict]  # [{source, page, text, score}, ...]
    used_context: bool  # did we find relevant chunks at all?


RAG_SYSTEM_PROMPT = """You are DocuMind, a precise technical documentation assistant.

Rules you must follow:
1. Answer the user's question using ONLY the provided context below.
2. If the context does not contain the answer, say "I don't have information \
about that in the provided documents." Do not use outside knowledge.
3. Cite your sources inline using [1], [2] etc. matching the numbered context.
4. Be concise and accurate. Do not pad the answer.
5. If you are uncertain, say so."""


def _build_context(results: list[SearchResult]) -> str:
    """Format retrieved chunks into a numbered context block for the prompt."""
    blocks = []
    for i, r in enumerate(results, start=1):
        page = r.chunk.extra.get("page", "?") if r.chunk.extra else "?"
        blocks.append(f"[{i}] (source: {r.chunk.source}, page: {page})\n{r.chunk.text}")
    return "\n\n".join(blocks)


class RAGPipeline:
    def __init__(self, retriever: Retriever):
        self.retriever = retriever
        self.llm = AsyncGroq(api_key=settings.groq_api_key)

    async def answer(
        self,
        question: str,
        k_final: int = 5,
        source_filter: str | None = None,
    ) -> RAGAnswer:
        # 1. Retrieve (hybrid search + rerank + threshold from Day 3)
        results = await self.retriever.retrieve(
            query=question,
            k_final=k_final,
            source_filter=source_filter,
        )

        # 2. No good chunks? Say so honestly — don't make things up.
        if not results or results[0].score < 0.15:
            return RAGAnswer(
                answer="I don't have information about that in the provided documents.",
                sources=[],
                used_context=False,
            )

        # 3. Build the augmented prompt
        context = _build_context(results)
        user_message = f"""Context:

{context}

---

Question: {question}

Answer using only the context above, citing sources with [n]."""

        # 4. Generate
        resp = await self.llm.chat.completions.create(
            model=settings.groq_model,
            messages=[
                {"role": "system", "content": RAG_SYSTEM_PROMPT},
                {"role": "user", "content": user_message},
            ],
            temperature=0.1,  # low — we want faithful, not creative
            max_tokens=1024,
        )
        answer_text = resp.choices[0].message.content or ""

        # 5. Package the sources so the UI can show them
        sources = [
            {
                "n": i + 1,
                "source": r.chunk.source,
                "page": r.chunk.extra.get("page", "?") if r.chunk.extra else "?",
                "score": round(r.score, 3),
                "text": r.chunk.text[:200],
            }
            for i, r in enumerate(results)
        ]

        return RAGAnswer(
            answer=answer_text,
            sources=sources,
            used_context=True,
        )
