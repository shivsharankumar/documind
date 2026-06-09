"""
Simple RAG evaluation using LLM-as-judge.
Builds the intuition behind RAGAS metrics by computing two of them by hand.
"""

import json

from groq import AsyncGroq

from documind.config import settings


class RAGEvaluator:
    def __init__(self):
        self.llm = AsyncGroq(api_key=settings.groq_api_key)
        # Use a strong model as the judge — judging needs good reasoning
        self.judge_model = settings.groq_model

    async def faithfulness(self, answer: str, context: str) -> dict:
        """
        Does the answer stick to the context, or did it make things up?
        Returns a score 0-1 and the reasoning.
        """
        prompt = f"""You are evaluating whether an AI answer is faithful to its source context.

CONTEXT:
{context}

ANSWER:
{answer}

Check every claim in the ANSWER. Is each claim supported by the CONTEXT?

Respond ONLY with JSON:
{{"score": <0.0 to 1.0>, "unsupported_claims": ["..."], "reasoning": "..."}}

score 1.0 = every claim supported by context
score 0.0 = answer contains claims not in context (hallucination)"""

        resp = await self.llm.chat.completions.create(
            model=self.judge_model,
            messages=[{"role": "user", "content": prompt}],
            temperature=0,
            response_format={"type": "json_object"},
        )
        return json.loads(resp.choices[0].message.content)

    async def answer_relevancy(self, question: str, answer: str) -> dict:
        """Does the answer actually address the question?"""
        prompt = f"""You are evaluating whether an answer addresses a question.

QUESTION: {question}

ANSWER: {answer}

Respond ONLY with JSON:
{{"score": <0.0 to 1.0>, "reasoning": "..."}}

score 1.0 = directly and fully answers the question
score 0.0 = does not address the question at all"""

        resp = await self.llm.chat.completions.create(
            model=self.judge_model,
            messages=[{"role": "user", "content": prompt}],
            temperature=0,
            response_format={"type": "json_object"},
        )
        return json.loads(resp.choices[0].message.content)
