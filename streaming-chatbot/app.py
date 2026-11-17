import asyncio
import json
import logging
import os
from collections.abc import AsyncIterator
from typing import Literal, Optional

from dotenv import load_dotenv
from fastapi import FastAPI, HTTPException, Request
from fastapi.middleware.cors import CORSMiddleware
from fastapi.responses import StreamingResponse
from openai import AsyncOpenAI, OpenAIError
from openai.types.responses import ResponseUsage
from pydantic import BaseModel, Field

load_dotenv()

# Without this, INFO-level messages (like client disconnects) are silently dropped.
logging.basicConfig(level=logging.INFO)
logger = logging.getLogger(__name__)

MODEL = os.getenv("OPENAI_MODEL", "gpt-4.1-mini")
MAX_HISTORY_MESSAGES = 12
MAX_MESSAGE_LENGTH = 10_000

CORS_ORIGINS = [
    "http://localhost:5173",
    "http://127.0.0.1:5173",
]

SSE_HEADERS = {
    "Cache-Control": "no-cache",
    "X-Accel-Buffering": "no",
}

SYSTEM_INSTRUCTIONS = """
You are a helpful AI engineering assistant.
Give accurate, clear and concise explanations.
Use practical examples when explaining technical concepts.
"""

app = FastAPI(title="AI Streaming Chatbot")

app.add_middleware(
    CORSMiddleware,
    allow_origins=CORS_ORIGINS,
    allow_credentials=False,
    allow_methods=["GET", "POST"],
    allow_headers=["Content-Type"],
)

client = AsyncOpenAI()


class ChatMessage(BaseModel):
    role: Literal["user", "assistant"]
    content: str = Field(min_length=1, max_length=MAX_MESSAGE_LENGTH)


class ChatRequest(BaseModel):
    message: str = Field(min_length=1, max_length=MAX_MESSAGE_LENGTH)
    history: list[ChatMessage] = Field(default_factory=list)


class TokenUsage(BaseModel):
    input_tokens: int
    output_tokens: int
    total_tokens: int


class ChatResponse(BaseModel):
    reply: str
    usage: Optional[TokenUsage] = None


def to_token_usage(usage: Optional[ResponseUsage]) -> Optional[TokenUsage]:
    if usage is None:
        return None

    return TokenUsage(
        input_tokens=usage.input_tokens,
        output_tokens=usage.output_tokens,
        total_tokens=usage.total_tokens,
    )


def build_model_input(payload: ChatRequest) -> list[dict[str, str]]:
    """Return the recent history followed by the new user message."""
    recent_history = [
        item.model_dump() for item in payload.history[-MAX_HISTORY_MESSAGES:]
    ]
    return [*recent_history, {"role": "user", "content": payload.message}]


def format_sse(payload: dict[str, object]) -> str:
    return f"data: {json.dumps(payload)}\n\n"


async def stream_events(
    http_request: Request,
    messages: list[dict[str, str]],
) -> AsyncIterator[str]:
    """Translate OpenAI response events into Server-Sent Events.

    Stops generating if the browser disconnects, so no tokens are paid for
    after the user has left.
    """
    try:
        async with await client.responses.create(
            model=MODEL,
            instructions=SYSTEM_INSTRUCTIONS,
            input=messages,
            stream=True,
        ) as stream:
            async for event in stream:
                if await http_request.is_disconnected():
                    logger.info("Client disconnected; stopping generation")
                    return

                if event.type == "response.output_text.delta":
                    yield format_sse({"type": "delta", "text": event.delta})

                elif event.type == "response.completed":
                    usage = to_token_usage(event.response.usage)
                    yield format_sse(
                        {
                            "type": "done",
                            "usage": usage.model_dump() if usage else None,
                        }
                    )
                    return

                elif event.type == "response.failed":
                    yield format_sse(
                        {"type": "error", "message": "Model generation failed."}
                    )
                    return

        yield format_sse(
            {"type": "error", "message": "Response ended unexpectedly."}
        )

    except asyncio.CancelledError:
        # Starlette cancels the generator when the client disconnects. Log it,
        # then re-raise so the cancellation completes.
        logger.info("Client disconnected; generation cancelled")
        raise

    except OpenAIError:
        logger.exception("OpenAI streaming request failed")
        yield format_sse(
            {"type": "error", "message": "AI service is temporarily unavailable."}
        )

    except Exception:
        logger.exception("Unexpected error while streaming")
        yield format_sse(
            {"type": "error", "message": "An unexpected error occurred."}
        )


@app.post("/chat/stream")
async def stream_chat(
    payload: ChatRequest,
    http_request: Request,
) -> StreamingResponse:
    return StreamingResponse(
        stream_events(http_request, build_model_input(payload)),
        media_type="text/event-stream",
        headers=SSE_HEADERS,
    )


@app.post("/chat", response_model=ChatResponse)
async def chat(payload: ChatRequest) -> ChatResponse:
    try:
        response = await client.responses.create(
            model=MODEL,
            instructions=SYSTEM_INSTRUCTIONS,
            input=build_model_input(payload),
        )
    except OpenAIError:
        logger.exception("OpenAI request failed")
        raise HTTPException(
            status_code=502,
            detail="AI service is temporarily unavailable.",
        )

    return ChatResponse(
        reply=response.output_text,
        usage=to_token_usage(response.usage),
    )


@app.get("/health")
def health_check() -> dict[str, str]:
    return {"status": "ok", "service": "ai-streaming-chatbot"}
