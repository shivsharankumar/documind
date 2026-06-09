import asyncio
import json
import logging
import time
from collections.abc import AsyncIterator

from fastapi import APIRouter, HTTPException, Request
from groq import APIError as GroqAPIError
from groq import RateLimitError as GroqRateLimit
from sse_starlette.sse import EventSourceResponse

from src.documind.api.schemas import ChatRequest, ChatResponse
from src.documind.config import settings

logger = logging.getLogger(__name__)
router = APIRouter()

SYSTEM_PROMPT = (
    "You are DocuMind, a precise technical assistant. "
    "If unsure, say so. Never make up code or APIs."
)


@router.post("/chat")
async def chat(req: ChatRequest, request: Request):
    """
    Main endpoint. Streams an answer back if req.stream=True,
    otherwise returns one big response.
    """
    if req.stream:
        return EventSourceResponse(_stream_chat(req, request))
    return await _chat_once(req, request)


async def _chat_once(req: ChatRequest, request: Request) -> ChatResponse:
    """Try Groq first; fall back to Gemini if Groq fails."""
    groq = request.app.state.groq
    gemini = request.app.state.gemini

    # Try Groq
    try:
        resp = await groq.chat.completions.create(
            model=settings.groq_model,
            messages=[
                {"role": "system", "content": SYSTEM_PROMPT},
                {"role": "user", "content": req.question},
            ],
            temperature=req.temperature,
            max_tokens=1024,
        )
        return ChatResponse(
            answer=resp.choices[0].message.content or "",
            input_tokens=resp.usage.prompt_tokens,
            output_tokens=resp.usage.completion_tokens,
            model=resp.model,
        )

    except (GroqRateLimit, GroqAPIError) as e:
        # Log so we KNOW the fallback happened
        logger.warning(f"Groq failed, falling back to Gemini: {e}")

        # Try Gemini
        try:
            resp = await gemini.aio.models.generate_content(
                model=settings.gemini_model,
                contents=f"{SYSTEM_PROMPT}\n\nUser: {req.question}",
                config={
                    "temperature": req.temperature,
                    "max_output_tokens": 1024,
                },
            )
            return ChatResponse(
                answer=resp.text or "",
                input_tokens=resp.usage_metadata.prompt_token_count,
                output_tokens=resp.usage_metadata.candidates_token_count,
                model=settings.gemini_model,
            )
        except Exception as gemini_err:
            # Both failed — give up cleanly
            logger.error(f"Both Groq and Gemini failed: groq={e}, gemini={gemini_err}")
            raise HTTPException(
                status_code=503,
                detail="All LLM providers unavailable. Try again later.",
            ) from gemini_err


async def _stream_chat(req: ChatRequest, request: Request) -> AsyncIterator[dict]:
    """The 'stream tokens as they come' version. This is the cool one."""
    groq = request.app.state.groq
    start = time.perf_counter()
    in_tokens = 0
    out_tokens = 0

    try:
        stream = await groq.chat.completions.create(
            model=settings.groq_model,
            messages=[
                {"role": "system", "content": SYSTEM_PROMPT},
                {"role": "user", "content": req.question},
            ],
            temperature=req.temperature,
            max_tokens=1024,
            stream=True,
        )

        async for chunk in stream:
            # CRITICAL: if user closed their browser, stop generating!
            # Otherwise we keep paying for tokens nobody will read.
            if await request.is_disconnected():
                break

            # Groq tells us token counts at the end of the stream
            if chunk.usage:
                in_tokens = chunk.usage.prompt_tokens
                out_tokens = chunk.usage.completion_tokens

            # Send the next piece of text to the user
            if chunk.choices:
                delta = chunk.choices[0].delta.content
                if delta:
                    yield {"event": "token", "data": delta}

        # Final message: tell them we're done, with stats
        elapsed_ms = int((time.perf_counter() - start) * 1000)
        yield {
            "event": "done",
            "data": json.dumps(
                {
                    "input_tokens": in_tokens,
                    "output_tokens": out_tokens,
                    "elapsed_ms": elapsed_ms,
                }
            ),
        }

    except asyncio.CancelledError:
        # User disconnected — clean shutdown
        raise
    except Exception as e:
        yield {"event": "error", "data": str(e)}
