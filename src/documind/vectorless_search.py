"""
Vectorless search: no embeddings, no vector DB.
Two approaches shown side by side.
Use when your document set is small (< 500 pages).
"""

import asyncio

from groq import AsyncGroq

from documind.config import settings

# ─── Approach A: Full Context (stuff everything in) ───────────────────────────


async def answer_with_full_context(
    question: str,
    documents: list[str],
    source_names: list[str],
) -> str:
    """
    No retrieval. Just give the LLM everything and ask it to answer.
    Simple. Surprisingly good for small collections.

    Limitation: all docs must fit in context window (~100K tokens max).
    Cost: you pay for ALL doc tokens on EVERY query.
    """
    # Format all docs
    formatted = "\n\n---\n\n".join(
        f"[Source: {name}]\n{doc}" for name, doc in zip(source_names, documents, strict=False)
    )

    prompt = f"""You have access to the following documents:

{formatted}

---

Answer this question based ONLY on the documents above.
If the answer isn't in the documents, say "I don't know based on the provided documents."

Question: {question}"""

    client = AsyncGroq(api_key=settings.groq_api_key)
    resp = await client.chat.completions.create(
        model=settings.groq_model,
        messages=[{"role": "user", "content": prompt}],
        temperature=0.1,
        max_tokens=1024,
    )
    return resp.choices[0].message.content or ""


# ─── Approach B: LLM-as-Retriever ─────────────────────────────────────────────


async def _is_relevant(
    client: AsyncGroq,
    question: str,
    doc: str,
    doc_name: str,
) -> tuple[bool, float]:
    """Ask the LLM: is this doc relevant to the question?"""
    prompt = f"""Question: {question}

Document ({doc_name}):
{doc[:500]}...

Is this document relevant to answering the question?
Reply with ONLY: YES or NO"""

    resp = await client.chat.completions.create(
        model="llama-3.1-8b-instant",  # use cheap model for filtering
        messages=[{"role": "user", "content": prompt}],
        temperature=0,
        max_tokens=5,
    )
    answer = (resp.choices[0].message.content or "").strip().upper()
    return answer == "YES", 1.0 if answer == "YES" else 0.0


async def answer_with_llm_retriever(
    question: str,
    documents: list[str],
    source_names: list[str],
) -> str:
    """
    LLM decides which documents are relevant, then answers from those.
    More accurate than vector search for nuanced relevance.
    Much more expensive — only for small critical collections.
    """
    client = AsyncGroq(api_key=settings.groq_api_key)

    # Filter docs in parallel (use cheap 8B model for yes/no)
    relevance_checks = await asyncio.gather(
        *(
            _is_relevant(client, question, doc, name)
            for doc, name in zip(documents, source_names, strict=False)
        )
    )

    relevant_docs = [
        (doc, name)
        for (doc, name), (is_rel, _) in zip(
            zip(documents, source_names, strict=False), relevance_checks, strict=False
        )
        if is_rel
    ]

    print(f"  LLM retriever: {len(relevant_docs)}/{len(documents)} docs relevant")

    if not relevant_docs:
        return "I couldn't find relevant information in the provided documents."

    # Now answer using only relevant docs
    docs_text = "\n\n---\n\n".join(f"[{name}]\n{doc}" for doc, name in relevant_docs)
    answer_prompt = f"""Based on these documents:

{docs_text}

Answer: {question}"""

    resp = await client.chat.completions.create(
        model=settings.groq_model,
        messages=[{"role": "user", "content": answer_prompt}],
        temperature=0.1,
        max_tokens=1024,
    )
    return resp.choices[0].message.content or ""
