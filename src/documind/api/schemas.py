from pydantic import BaseModel, Field


class ChatRequest(BaseModel):
    question: str = Field(min_length=1, max_length=4000)
    temperature: float = Field(default=0.2, ge=0.0, le=2.0)
    stream: bool = True


class ChatResponse(BaseModel):
    answer: str
    input_tokens: int
    output_tokens: int
    model: str


class ErrorResponse(BaseModel):
    error: str
    detail: str | None = None
