"""Evaluate DocuMind's RAG quality on a set of test questions."""

import asyncio
import sys
from pathlib import Path

sys.path.append(str(Path(__file__).resolve().parent.parent / "src"))
from documind.evaluate import RAGEvaluator
from documind.rag import RAGPipeline
from documind.retriever import Retriever

# Your "golden" test set — questions you know the docs can answer.
# In production this grows to 50-200 questions covering edge cases.
# scripts/04_eval.py — update TEST_QUESTIONS with these

TEST_QUESTIONS = [
    # ── 5 questions the paper CAN answer ──────────────────────────
    "What is self-attention?",
    "What is the Transformer architecture?",
    "What are the components of multi-head attention?",
    "Why did the authors move away from recurrent neural networks?",
    "What BLEU score did the Transformer achieve on WMT 2014 English-to-German?",
    # ── 2 questions the paper CANNOT answer ───────────────────────
    "What is the capital of France?",
    "How do I install Python on Windows?",
]


async def main():
    retriever = Retriever()
    await retriever.setup()
    rag = RAGPipeline(retriever)
    evaluator = RAGEvaluator()

    faithfulness_scores = []
    relevancy_scores = []

    for q in TEST_QUESTIONS:
        result = await rag.answer(q)

        if not result.used_context:
            print(f"Q: {q}\n  → No context found (skipping)\n")
            continue

        # Rebuild the context string that was used (for the judge)
        context = "\n\n".join(s["text"] for s in result.sources)

        faith = await evaluator.faithfulness(result.answer, context)
        rel = await evaluator.answer_relevancy(q, result.answer)

        faithfulness_scores.append(faith["score"])
        relevancy_scores.append(rel["score"])

        print(f"Q: {q}")
        print(f"  Answer: {result.answer[:100]}...")
        print(f"  Faithfulness: {faith['score']:.2f}")
        print(f"  Relevancy:    {rel['score']:.2f}")
        if faith["score"] < 1.0:
            print(f"  ⚠️  Unsupported: {faith.get('unsupported_claims')}")
        print()

    if faithfulness_scores:
        print("─" * 40)
        print(f"Avg Faithfulness: {sum(faithfulness_scores) / len(faithfulness_scores):.2f}")
        print(f"Avg Relevancy:    {sum(relevancy_scores) / len(relevancy_scores):.2f}")


if __name__ == "__main__":
    asyncio.run(main())
