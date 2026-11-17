
from fastapi import FastAPI
from fastapi import HTTPException
from openai import OpenAI, OpenAIError
from pydantic import BaseModel, Field
from typing import Literal
from collections.abc import Iterator
from fastapi.responses import StreamingResponse

import json
import os


app = FastAPI(title="AI Streaming Chatbot")

client = OpenAI()

MODEL = os.getenv("OPENAI_MODEL", "gpt-4.1-mini")

SYSTEM_INSTRUCTIONS = """
You are a helpful AI engineering assistant.
Give accurate, clear and concise explanations.
Use practical examples when explaining technical concepts.
"""


class ChatMessage(BaseModel):
    role: Literal["user", "assistant"]
    content: str = Field(min_length=1, max_length=10000)


class ChatRequest(BaseModel):
    message: str = Field(min_length=1, max_length=10000)
    history: list[ChatMessage] = Field(default_factory=list)


def format_sse(payload: dict) -> str:
    return f"data: {json.dumps(payload)}\n\n"


@app.post("/chat/stream")
def stream_chat(request: ChatRequest):

    messages = [
        *[
            item.model_dump()
            for item in request.history[-12:]
        ],
        {
            "role": "user",
            "content": request.message
        }
    ]

    def generate() -> Iterator[str]:
        try:
            stream = client.responses.create(
                model=MODEL,
                instructions=SYSTEM_INSTRUCTIONS,
                input=messages,
                stream=True,
            )

            completed = False

            for event in stream:
                if event.type == "response.output_text.delta":
                    yield format_sse({
                        "type": "delta",
                        "text": event.delta
                    })

                elif event.type == "response.completed":
                    completed = True

                elif event.type == "response.failed":
                    yield format_sse({
                        "type": "error",
                        "message": "Model generation failed."
                    })
                    return

            if not completed:
                yield format_sse({
                    "type": "error",
                    "message": "Response ended unexpectedly."
                })
                return

            yield format_sse({"type": "done"})

        except OpenAIError:
            yield format_sse({
                "type": "error",
                "message": "AI service is temporarily unavailable."
            })

    return StreamingResponse(
        generate(),
        media_type="text/event-stream",
        headers={
            "Cache-Control": "no-cache",
            "X-Accel-Buffering": "no"
        }
    )


@app.post("/chat")
def chat(request: ChatRequest):
    messages = [
        *[
            item.model_dump()
            for item in request.history[-12:]
        ],
        {
            "role": "user",
            "content": request.message
        }
    ]

    try:
        response = client.responses.create(
            model=MODEL,
            instructions=SYSTEM_INSTRUCTIONS,
            input=messages
        )

        return {
            "reply": response.output_text
        }

    except OpenAIError:
        raise HTTPException(
            status_code=502,
            detail="AI service is temporarily unavailable."
        )

@app.get("/health")
def health_check():
    return {
        "status": "ok",
        "service": "ai-streaming-chatbot"
    }