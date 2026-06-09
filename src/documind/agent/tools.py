"""
Agent tools — functions the agent can call.
Each tool has a name, description (what the LLM reads to decide
when to use it), and an async run() method.
"""

from dataclasses import dataclass
from typing import Any

from documind.rag import RAGPipeline
from documind.retriever import Retriever


@dataclass
class ToolResult:
    """Structured result from any tool call."""

    tool_name: str
    success: bool
    output: str  # what the agent sees in its context
    raw: Any = None  # the full Python object, for downstream use


class SearchDocsTool:
    name = "search_docs"
    description = (
        "Search the document store for relevant chunks. "
        "Use this when you need to find specific information from the documents. "
        "Input: a search query string. "
        "Output: top relevant text chunks with source and page."
    )

    def __init__(self, retriever: Retriever):
        self.retriever = retriever

    async def run(self, query: str) -> ToolResult:
        results = await self.retriever.retrieve(query, k_final=4)
        if not results:
            return ToolResult(
                tool_name=self.name,
                success=False,
                output="No relevant documents found for this query.",
            )
        # Format for the LLM to read
        formatted = "\n\n".join(
            f"[{i + 1}] (source: {r.chunk.source}, "
            f"page: {r.chunk.extra.get('page', '?') if r.chunk.extra else '?'}, "
            f"score: {r.score:.2f})\n{r.chunk.text}"
            for i, r in enumerate(results)
        )
        return ToolResult(
            tool_name=self.name,
            success=True,
            output=formatted,
            raw=results,
        )


class AskRAGTool:
    name = "ask_rag"
    description = (
        "Get a complete cited answer to a question using RAG. "
        "Use this when you want a synthesized answer, not just raw chunks. "
        "Input: a specific question. "
        "Output: a cited answer with source references."
    )

    def __init__(self, rag: RAGPipeline):
        self.rag = rag

    async def run(self, question: str) -> ToolResult:
        result = await self.rag.answer(question)
        if not result.used_context:
            return ToolResult(
                tool_name=self.name,
                success=False,
                output="Could not find relevant information to answer this question.",
            )
        sources_str = ", ".join(f"[{s['n']}] {s['source']} p.{s['page']}" for s in result.sources)
        return ToolResult(
            tool_name=self.name,
            success=True,
            output=f"{result.answer}\n\nSources: {sources_str}",
            raw=result,
        )


class CheckAnswerTool:
    """
    The agent's self-critic. Before giving a final answer,
    the agent checks: is my answer actually grounded in evidence?
    This is a lightweight version of the faithfulness eval from Day 4.
    """

    name = "check_answer"
    description = (
        "Verify that a proposed answer is grounded in retrieved evidence. "
        "Use this before giving a final answer to catch hallucinations. "
        "Input: 'answer|||evidence' (answer and evidence separated by |||). "
        "Output: PASS or FAIL with reasoning."
    )

    def __init__(self, llm_client):
        self.llm = llm_client

    async def run(self, input_str: str) -> ToolResult:
        if "|||" not in input_str:
            return ToolResult(
                tool_name=self.name,
                success=False,
                output="FAIL: Input must be 'answer|||evidence'",
            )

        answer, evidence = input_str.split("|||", 1)

        prompt = f"""Is this answer grounded in the evidence?
Check if every claim in the answer is supported by the evidence.

EVIDENCE:
{evidence.strip()}

ANSWER:
{answer.strip()}

Reply with exactly: PASS or FAIL, then one sentence explaining why."""

        resp = await self.llm.chat.completions.create(
            model="llama-3.1-8b-instant",  # cheap model for self-check
            messages=[{"role": "user", "content": prompt}],
            temperature=0,
            max_tokens=100,
        )
        verdict = resp.choices[0].message.content or "FAIL: no response"

        return ToolResult(
            tool_name=self.name,
            success=verdict.strip().startswith("PASS"),
            output=verdict,
        )


class FinishTool:
    """Signal that the agent is done and return the final answer."""

    name = "finish"
    description = (
        "Return the final answer to the user. "
        "Use this when you have a complete, verified answer. "
        "Input: the final answer string."
    )

    async def run(self, answer: str) -> ToolResult:
        return ToolResult(
            tool_name=self.name,
            success=True,
            output=answer,
        )
