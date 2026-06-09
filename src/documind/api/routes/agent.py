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
