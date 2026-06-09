"""Watch the ReAct agent reason step by step."""

import asyncio
import sys
from pathlib import Path

sys.path.insert(0, str(Path(__file__).parent.parent / "src"))

from documind.agent.react_agent import ReActAgent
from documind.rag import RAGPipeline
from documind.retriever import Retriever


async def main():
    # Setup
    retriever = Retriever()
    await retriever.setup()
    rag = RAGPipeline(retriever)
    agent = ReActAgent(retriever=retriever, rag_pipeline=rag)

    # Test questions — these require multi-step reasoning
    questions = [
        # Simple — should take 2-3 steps
        # "What is self-attention according to the Transformer paper?",
        # Medium — requires two searches and synthesis
        "What are the advantages of self-attention over recurrent layers, "
        "and what task did the Transformer achieve state-of-the-art on?",
        # Hard — requires reasoning across multiple pieces
        "How does the Transformer handle sequence order, and why is this "
        "approach better than RNNs for this purpose?",
    ]

    for q in questions:
        result = await agent.run(q)
        print(f"\n{'─' * 60}")
        print(f"FINAL ANSWER:\n{result.answer}")
        print(f"Steps taken: {result.total_steps}")
        print(f"Success: {result.success}")
        print("─" * 60)


if __name__ == "__main__":
    asyncio.run(main())
