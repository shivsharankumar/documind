"""
ReAct agent — Reasoning + Acting loop.
Built from scratch so every step is visible and debuggable.
"""

import re
from dataclasses import dataclass

from groq import AsyncGroq

from documind.agent.tools import AskRAGTool, CheckAnswerTool, FinishTool, SearchDocsTool, ToolResult
from documind.config import settings


@dataclass
class AgentStep:
    """One step in the agent's reasoning trace."""

    thought: str
    action: str  # tool name
    action_input: str  # what was passed to the tool
    observation: str  # what the tool returned
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
            f"- {tool.name}: {tool.description}" for tool in self.tools.values()
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
        action_match = re.search(r"Action\s*:\s*(.+?)(?:\n|$)", text, re.IGNORECASE)
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
                output=(f"Unknown tool '{action}'. Available tools: {list(self.tools.keys())}"),
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

        print(f"\n{'=' * 60}")
        print(f"Agent starting: {question}")
        print(f"{'=' * 60}")

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
            steps.append(
                AgentStep(
                    thought=thought,
                    action=action,
                    action_input=action_input,
                    observation=result.output,
                    step_number=step_num,
                )
            )

            # ── Add the step to conversation history ──────────────────
            # This is how the agent "remembers" what it's done
            messages.append(
                {
                    "role": "assistant",
                    "content": llm_output,
                }
            )
            messages.append(
                {
                    "role": "user",
                    "content": f"Observation: {result.output}",
                }
            )

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
