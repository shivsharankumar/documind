# Day 5: Agentic AI Foundation

## Where we are

So far DocuMind can:
- Ingest PDFs and store them in Qdrant
- Find relevant chunks using hybrid search + reranking
- Generate cited answers using those chunks

But it's still **reactive** — it only does exactly what you tell it, one step at a time. You ask, it answers. Done.

An **agent** is different. You give it a *goal*, and it figures out the *steps* to reach that goal by itself. It can use tools, check its own work, try again if something failed, and chain multiple actions together.

Today we build DocuMind's first agent from scratch. By end of day it will:
1. Understand what a ReAct agent is and why it works
2. Have real tools it can call (search docs, answer from RAG, check its own answer)
3. Run a reasoning loop — think → act → observe → think again
4. Handle the reliability problems agents introduce in production

No LangGraph yet (that's Day 6). Today we build the agent loop **by hand** so you understand exactly what frameworks like LangGraph are automating. Same philosophy as Day 4 — understand it naked before you put clothes on it.

---

## Part 1: What an agent actually is

Let me kill the hype first.

An agent is not magic. It's not "AI that thinks." It's a **loop** with three things:

```
┌─────────────────────────────────────────────┐
│              THE AGENT LOOP                  │
│                                              │
│   1. LLM thinks: "what should I do next?"   │
│            │                                 │
│            ▼                                 │
│   2. LLM calls a tool (or says "done")       │
│            │                                 │
│            ▼                                 │
│   3. Tool runs, result comes back            │
│            │                                 │
│            ▼                                 │
│   4. Result added to context, go to 1        │
└─────────────────────────────────────────────┘
```

That's literally it. The LLM decides what tool to call, the tool runs, the result goes back to the LLM, repeat. The "intelligence" is entirely in the LLM deciding what to do next based on what it's seen so far.

### The restaurant analogy

Your RAG pipeline from Day 4 is like a **waiter who takes your order and brings exactly what you asked for.** You say "pasta," they bring pasta.

An agent is like a **personal chef.** You say "I'm hungry and have a dinner party in 2 hours." The chef:
1. Thinks: "what do I need to know? How many people? Any dietary restrictions?"
2. Asks you those questions (tool: communicate)
3. Thinks: "I'll make pasta. Let me check what ingredients we have"
4. Checks the fridge (tool: search_inventory)
5. Discovers no tomatoes
6. Thinks: "I need tomatoes. Let me order them"
7. Orders groceries (tool: order_groceries)
8. Cooks the meal
9. Serves you

The same goal — "feed the dinner party" — but the chef planned the steps, used multiple tools, recovered from a missing ingredient, and reached the goal without you micromanaging every step.

---

## Part 2: ReAct — the pattern that makes agents work

**ReAct** = **Re**asoning + **Act**ing. Published as a research paper in 2022, it's become the standard pattern for LLM agents.

The core insight: **if you ask the LLM to think out loud before acting, it makes better decisions.**

Without ReAct:
```
User: "What does the Transformer paper say about attention, and is that
       consistent with how BERT works?"
Agent: [immediately calls search tool with "attention BERT"]
       → often wrong or incomplete
```

With ReAct:
```
User: "What does the Transformer paper say about attention, and is that
       consistent with how BERT works?"

Thought: The user is asking two things: (1) what the Transformer paper
         says about attention, and (2) whether that's consistent with BERT.
         I should first search for what the Transformer paper says about
         attention, then search for BERT's attention mechanism, then compare.

Action: search_docs("Transformer paper attention mechanism")
Observation: "Self-attention relates different positions of a single
              sequence to compute a representation..."

Thought: Good. Now I need to find how BERT uses attention.
Action: search_docs("BERT attention mechanism")
Observation: "BERT uses bidirectional self-attention across all layers..."

Thought: I have both. BERT's attention IS based on the Transformer's
         self-attention. They're consistent — BERT adopted the mechanism.
Final Answer: [synthesized answer with citations]
```

The Thought step is what makes this work. It forces the LLM to plan before acting, which dramatically reduces wrong tool calls and wasted steps.

In code, this looks like a structured conversation where the LLM outputs either a `Thought+Action` pair or a `Final Answer`.

---

## Part 3: Tools — giving the agent hands

A tool is just a Python function with a clear description. The LLM reads the description and decides when to call it. That's the whole interface.

For DocuMind's agent, we'll build four tools:

| Tool | What it does | When agent uses it |
|---|---|---|
| `search_docs` | Hybrid search + rerank in Qdrant | Finding relevant document chunks |
| `ask_rag` | Full RAG answer from a question | Getting a cited answer |
| `check_answer` | Verifies if answer is grounded | Self-checking before responding |
| `finish` | Returns the final answer | When the agent is confident |

Let's define them with a clean interface. `src/documind/agent/tools.py`:

```python
"""
Agent tools — functions the agent can call.
Each tool has a name, description (what the LLM reads to decide
when to use it), and an async run() method.
"""
from dataclasses import dataclass
from typing import Any
from documind.retriever import Retriever
from documind.rag import RAGPipeline


@dataclass
class ToolResult:
    """Structured result from any tool call."""
    tool_name: str
    success: bool
    output: str        # what the agent sees in its context
    raw: Any = None    # the full Python object, for downstream use


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
            f"[{i+1}] (source: {r.chunk.source}, "
            f"page: {r.chunk.extra.get('page','?') if r.chunk.extra else '?'}, "
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
        sources_str = ", ".join(
            f"[{s['n']}] {s['source']} p.{s['page']}"
            for s in result.sources
        )
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
```

---

## Part 4: The ReAct agent loop

Now the heart of it. The agent loop is a conversation between us and the LLM where the LLM's "turns" are structured as `Thought + Action` pairs.

`src/documind/agent/react_agent.py`:

```python
"""
ReAct agent — Reasoning + Acting loop.
Built from scratch so every step is visible and debuggable.
"""
import json
import re
from dataclasses import dataclass, field
from typing import Any
from groq import AsyncGroq
from documind.config import settings
from documind.agent.tools import (
    SearchDocsTool, AskRAGTool, CheckAnswerTool, FinishTool, ToolResult
)


@dataclass
class AgentStep:
    """One step in the agent's reasoning trace."""
    thought: str
    action: str          # tool name
    action_input: str    # what was passed to the tool
    observation: str     # what the tool returned
    step_number: int


@dataclass
class AgentResult:
    """Final result from the agent."""
    answer: str
    steps: list[AgentStep]
    success: bool
    total_steps: int


# ── The system prompt is the agent's "brain wiring" ──────────────────────────
REACT_SYSTEM_PROMPT = """You are DocuMind Agent, an AI that answers questions
about documents by reasoning step by step and using tools.

You have access to these tools:
{tool_descriptions}

ALWAYS follow this exact format for EVERY response:

Thought: [your reasoning about what to do next]
Action: [tool_name]
Action Input: [input to the tool]

When you have a final answer ready:
Thought: [your reasoning that the answer is complete and verified]
Action: finish
Action Input: [your complete final answer with citations]

RULES:
- Always think before acting
- Use search_docs to find information, ask_rag for synthesized answers
- Use check_answer to verify before finishing
- Never make up information not found in the documents
- If documents don't contain the answer, say so in your finish action
- Maximum {max_steps} steps — be efficient"""


class ReActAgent:
    def __init__(
        self,
        retriever,
        rag_pipeline,
        max_steps: int = 8,
    ):
        self.llm = AsyncGroq(api_key=settings.groq_api_key)
        self.max_steps = max_steps

        # Register tools
        self.tools = {
            "search_docs": SearchDocsTool(retriever),
            "ask_rag": AskRAGTool(rag_pipeline),
            "check_answer": CheckAnswerTool(self.llm),
            "finish": FinishTool(),
        }

        # Build the tool descriptions the LLM sees
        self.tool_descriptions = "\n".join(
            f"- {tool.name}: {tool.description}"
            for tool in self.tools.values()
        )

    def _build_system_prompt(self) -> str:
        return REACT_SYSTEM_PROMPT.format(
            tool_descriptions=self.tool_descriptions,
            max_steps=self.max_steps,
        )

    def _parse_action(self, text: str) -> tuple[str, str]:
        """
        Parse the LLM's Thought/Action/Action Input response.
        Returns (action_name, action_input).

        The LLM sometimes formats things slightly wrong — we handle
        the common variations rather than crashing.
        """
        # Extract Action
        action_match = re.search(
            r"Action\s*:\s*(.+?)(?:\n|$)", text, re.IGNORECASE
        )
        # Extract Action Input
        input_match = re.search(
            r"Action\s*Input\s*:\s*(.+?)(?:\nThought|\nAction|$)",
            text,
            re.IGNORECASE | re.DOTALL,
        )
        # Extract Thought
        thought_match = re.search(
            r"Thought\s*:\s*(.+?)(?:\nAction|$)",
            text,
            re.IGNORECASE | re.DOTALL,
        )

        action = action_match.group(1).strip() if action_match else "finish"
        action_input = input_match.group(1).strip() if input_match else text
        thought = thought_match.group(1).strip() if thought_match else ""

        # Normalize action name — LLM sometimes adds spaces or caps
        action = action.lower().replace(" ", "_")

        return action, action_input, thought

    async def _call_tool(self, action: str, action_input: str) -> ToolResult:
        """Call a tool by name. Handle unknown tools gracefully."""
        if action not in self.tools:
            # LLM hallucinated a tool name — tell it that
            return ToolResult(
                tool_name=action,
                success=False,
                output=(
                    f"Unknown tool '{action}'. "
                    f"Available tools: {list(self.tools.keys())}"
                ),
            )
        return await self.tools[action].run(action_input)

    async def run(self, question: str) -> AgentResult:
        """
        Run the ReAct loop until the agent finishes or hits max_steps.
        """
        steps: list[AgentStep] = []

        # The conversation history — grows with each step
        messages = [
            {"role": "system", "content": self._build_system_prompt()},
            {"role": "user", "content": question},
        ]

        print(f"\n{'='*60}")
        print(f"Agent starting: {question}")
        print(f"{'='*60}")

        for step_num in range(1, self.max_steps + 1):
            print(f"\n[Step {step_num}]")

            # ── LLM thinks and decides what to do ────────────────────
            resp = await self.llm.chat.completions.create(
                model=settings.groq_model,
                messages=messages,
                temperature=0.1,
                max_tokens=512,
            )
            llm_output = resp.choices[0].message.content or ""
            print(f"LLM output:\n{llm_output}")

            # ── Parse the thought + action ────────────────────────────
            action, action_input, thought = self._parse_action(llm_output)

            # ── If agent says finish — we're done ─────────────────────
            if action == "finish":
                print(f"\n✅ Agent finished after {step_num} steps")
                return AgentResult(
                    answer=action_input,
                    steps=steps,
                    success=True,
                    total_steps=step_num,
                )

            # ── Run the tool ──────────────────────────────────────────
            print(f"Calling tool: {action}({action_input[:80]}...)")
            result = await self._call_tool(action, action_input)
            print(f"Tool result: {result.output[:150]}...")

            # ── Record this step ──────────────────────────────────────
            steps.append(AgentStep(
                thought=thought,
                action=action,
                action_input=action_input,
                observation=result.output,
                step_number=step_num,
            ))

            # ── Add the step to conversation history ──────────────────
            # This is how the agent "remembers" what it's done
            messages.append({
                "role": "assistant",
                "content": llm_output,
            })
            messages.append({
                "role": "user",
                "content": f"Observation: {result.output}",
            })

        # ── Hit max steps without finishing ───────────────────────────
        print(f"\n⚠️  Agent hit max steps ({self.max_steps})")
        return AgentResult(
            answer=(
                "I was unable to fully answer your question within "
                f"the step limit ({self.max_steps} steps). "
                "Try asking a more specific question."
            ),
            steps=steps,
            success=False,
            total_steps=self.max_steps,
        )
```

The most important line in that whole file is this one, in the loop:

```python
messages.append({"role": "user", "content": f"Observation: {result.output}"})
```

**This is how agents work.** The tool result goes back into the conversation as a new message. The LLM's next "turn" sees everything that happened before — the original question, every thought, every action, every observation — and uses that full context to decide what to do next. There's no magic memory. It's all just the growing conversation history.

---

## Part 5: A critical production problem — the growing context

Here's something the course won't tell you until it bites you:

**Every step adds tokens to the conversation.** After 5 steps, the LLM is reading the entire history — original question + 5 thoughts + 5 tool calls + 5 observations. After 10 steps that's a lot of tokens. Three problems:

```
Step 1:  ~500 tokens    → cheap, fast
Step 5:  ~3,000 tokens  → noticeable cost
Step 10: ~8,000 tokens  → getting expensive
Step 20: ~20,000 tokens → slow, very expensive
```

This is why `max_steps` exists. It's not just a safety net against infinite loops — it's a **cost budget**. In production you'll also add:

- **Token counting per step** — if the context exceeds a limit, summarize earlier steps
- **Step timeouts** — each tool call has a deadline
- **Total cost budget** — if this agent call would cost > $0.50, abort

We'll add proper observability for this on Day 7 (LangSmith). For now, `max_steps=8` is a reasonable limit.

---

## Part 6: Wire the agent into the API

Add an agent endpoint. `src/documind/api/routes/agent.py`:

```python
from fastapi import APIRouter, Request
from pydantic import BaseModel, Field

router = APIRouter()


class AgentRequest(BaseModel):
    question: str = Field(min_length=1, max_length=4000)


@router.post("/agent")
async def agent_endpoint(req: AgentRequest, request: Request):
    agent = request.app.state.agent
    result = await agent.run(req.question)

    return {
        "answer": result.answer,
        "success": result.success,
        "total_steps": result.total_steps,
        "trace": [
            {
                "step": s.step_number,
                "thought": s.thought,
                "action": s.action,
                "action_input": s.action_input[:200],
                "observation": s.observation[:300],
            }
            for s in result.steps
        ],
    }
```

Notice we return the full `trace` — every thought and action. This is not just for debugging. In a real product, showing users "here's how the agent reasoned" builds trust. Users are more likely to trust an answer they can see was derived step by step than one that appeared from a black box.

Update `app.py` lifespan:

```python
from documind.agent.react_agent import ReActAgent
from documind.api.routes import agent as agent_route

@asynccontextmanager
async def lifespan(app: FastAPI):
    app.state.groq = AsyncGroq(api_key=settings.groq_api_key)
    app.state.gemini = genai.Client(api_key=settings.gemini_api_key)

    retriever = Retriever()
    await retriever.setup()

    rag = RAGPipeline(retriever)
    app.state.rag = rag

    # Agent gets access to both retriever and RAG pipeline
    app.state.agent = ReActAgent(retriever=retriever, rag_pipeline=rag)

    yield
    await app.state.groq.close()

app.include_router(chat.router, prefix="/v1")
app.include_router(ask.router, prefix="/v1")
app.include_router(agent_route.router, prefix="/v1")
```

---

## Part 7: Test script — watch the agent think

`scripts/05_test_agent.py`:

```python
"""Watch the ReAct agent reason step by step."""
import asyncio
import sys
from pathlib import Path
sys.path.insert(0, str(Path(__file__).parent.parent / "src"))

from documind.retriever import Retriever
from documind.rag import RAGPipeline
from documind.agent.react_agent import ReActAgent


async def main():
    # Setup
    retriever = Retriever()
    await retriever.setup()
    rag = RAGPipeline(retriever)
    agent = ReActAgent(retriever=retriever, rag_pipeline=rag)

    # Test questions — these require multi-step reasoning
    questions = [
        # Simple — should take 2-3 steps
        "What is self-attention according to the Transformer paper?",

        # Medium — requires two searches and synthesis
        "What are the advantages of self-attention over recurrent layers, "
        "and what task did the Transformer achieve state-of-the-art on?",

        # Hard — requires reasoning across multiple pieces
        "How does the Transformer handle sequence order, and why is this "
        "approach better than RNNs for this purpose?",
    ]

    for q in questions:
        result = await agent.run(q)
        print(f"\n{'─'*60}")
        print(f"FINAL ANSWER:\n{result.answer}")
        print(f"Steps taken: {result.total_steps}")
        print(f"Success: {result.success}")
        print("─"*60)


if __name__ == "__main__":
    asyncio.run(main())
```

Run it:
```bash
uv run python scripts/05_test_agent.py
```

Watch the output carefully. You should see the agent:
1. Thinking about what it needs to do
2. Calling `search_docs` or `ask_rag`
3. Reading the observation
4. Thinking again based on what it found
5. Calling `check_answer` before finishing
6. Returning a final answer

That trace is the agent's reasoning made visible. Every professional agent system shows you this trace — LangSmith (Day 7) records it automatically.

---

## Part 8: Reliability caveats — what agents get wrong in production

The course lists "reliability caveats" as a Day 5 topic. This is the most important part of agents that tutorials skip. Let me be direct.

### Caveat 1: Agents are non-deterministic

The same question can take completely different paths on different runs. Sometimes 3 steps, sometimes 7. Sometimes it finds the answer immediately; sometimes it searches in circles. This makes testing hard — you can't `assertEqual` the trace, only the final answer quality.

### Caveat 2: Tool call parsing fails

The LLM is supposed to output:
```
Thought: I need to search for this
Action: search_docs
Action Input: transformer attention mechanism
```

But sometimes it outputs:
```
I'll search for this using search_docs with the query "transformer attention"
```

Or:
```
Action: Search Docs  ← capitalized, with space
Action Input: "transformer attention"  ← with quotes
```

Our parser handles some variations but not all. In production you need:
- Robust regex that handles common LLM formatting variations
- Fallback behavior when parsing fails (ask the LLM to retry)
- Logging of every parse failure so you can improve the prompt

### Caveat 3: Agents can loop

Without `max_steps`, an agent can call the same tool repeatedly with slightly different queries, never getting closer to an answer. Always set `max_steps`. Always.

### Caveat 4: Costs compound

Every step = one LLM call. A 6-step agent run = 6 LLM calls. For 1000 users, that's 6000 LLM calls per minute at peak. Your rate limits will hit. Your costs will surprise you. This is why Day 7 (observability) tracks per-agent-run costs.

### Caveat 5: The agent doesn't know what it doesn't know

If the documents don't contain the answer, a well-prompted agent says "I don't know." A poorly-prompted one searches 8 times with different queries hoping to find something, runs out of steps, and returns a confusing non-answer. The `max_steps` error message is your safety net — make it informative.

### The honest assessment of agents in 2026

Agents work reliably for **narrow, well-defined tasks** with good tools. They work unreliably for **open-ended tasks** with many possible paths. DocuMind's agent is narrow (answer questions from documents) with good tools (search, RAG, verify). That's the sweet spot.

For anything requiring more than ~10 steps, human-in-the-loop becomes necessary. We'll cover that on Day 8 (Evals & Guardrails in Production).

---

## Day 5 exercises

**Exercise 5.1 — Run the agent**
Run `scripts/05_test_agent.py`. Paste the trace for the medium question ("advantages of self-attention over recurrent layers"). Count the steps. Did it use `check_answer` before finishing? Did the final answer cite sources?

**Exercise 5.2 — Watch it fail**
Ask the agent a question your PDF can't answer:
```bash
curl -X POST http://localhost:8000/v1/agent \
  -H "content-type: application/json" \
  -d '{"question": "What is the GDP of India in 2023?"}'
```
Watch the trace. How many steps does it take before it gives up? Does it say "I don't know" or does it hallucinate? This tells you how robust your prompt is.

**Exercise 5.3 — Break the parser (then fix it)**
Temporarily change the system prompt to NOT require the `Action:` and `Action Input:` format — just say "use the tools as needed." Watch `_parse_action` fail. Then restore the format requirement. This shows you why the structured output format in the prompt is not optional.

**Exercise 5.4 — Add a new tool**
Add a `summarize_topic` tool that takes a topic string, searches for relevant chunks, and returns a 3-sentence summary. Register it in `ReActAgent.__init__`. Test that the agent can discover and use it for broad questions like "summarize what the paper says about multi-head attention."

**Exercise 5.5 — Explain it back**
In your own words:
1. What is the difference between a RAG pipeline and an agent?
2. Why does the agent need to see the full conversation history on every step?
3. What is `max_steps` protecting against (give two answers: correctness and cost)?
4. Why do we use a cheap model (`llama-3.1-8b-instant`) for `check_answer` instead of the main model?

---

## What you should walk away with

```
RAG (Day 4)         Agent (Day 5)
─────────────────   ──────────────────────────────
One fixed path      Dynamic path decided at runtime
One LLM call        Multiple LLM calls in a loop
You plan the steps  Agent plans the steps
Predictable cost    Variable cost (bounded by max_steps)
Fast                Slower (each step = one LLM call)
```

Neither is better — they serve different needs. RAG for predictable Q&A; agents for tasks requiring planning, multi-step reasoning, or tool use.

**The three things to remember from today:**
1. An agent is just a loop — think → act → observe → repeat
2. ReAct (thinking before acting) is what makes agents reliable
3. `max_steps` is both a safety net AND a cost budget — never omit it

---

Say **"next"** for **Day 6: Agent Orchestration & LangGraph** — where we graduate from a single agent to a proper workflow. We'll build DocuMind's research pipeline as a state machine with conditional branches, parallel execution, and proper error recovery. LangGraph is the tool; the concept is workflow orchestration.

Or if you want the Day 5 notes `.md` file first, just say so.
