from fastapi import APIRouter, Request
from pydantic import BaseModel, Field

router = APIRouter()


class AskRequest(BaseModel):
    question: str = Field(min_length=1, max_length=4000)
    source: str | None = None  # optional: restrict to one document


@router.post("/ask")
async def ask(req: AskRequest, request: Request):
    rag = request.app.state.rag
    result = await rag.answer(req.question, source_filter=req.source)
    return {
        "answer": result.answer,
        "sources": result.sources,
        "used_context": result.used_context,
    }
