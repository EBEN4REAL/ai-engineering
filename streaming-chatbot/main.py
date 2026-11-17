import os

from dotenv import load_dotenv
from openai import OpenAI, OpenAIError

load_dotenv()

MODEL = os.getenv("OPENAI_MODEL", "gpt-4.1-mini")
MAX_HISTORY_MESSAGES = 12

SYSTEM_INSTRUCTIONS = """
You are a helpful AI assistant.

Be accurate, clear, and concise.
Specialise in explaining software engineering and AI concepts.
If you don't know something, acknowledge the uncertainty.
"""

EXIT_COMMANDS = {"/exit", "exit", "quit"}
CLEAR_COMMANDS = {"/clear", "/reset"}

client = OpenAI()

Message = dict[str, str]


def stream_reply(messages: list[Message]) -> str | None:
    """Print the streamed reply and return it, or None if the request failed."""
    parts: list[str] = []

    try:
        stream = client.responses.create(
            model=MODEL,
            instructions=SYSTEM_INSTRUCTIONS,
            input=messages,
            stream=True,
        )

        for event in stream:
            if event.type == "response.output_text.delta":
                print(event.delta, end="", flush=True)
                parts.append(event.delta)

            elif event.type == "response.completed":
                print()
                return "".join(parts)

            elif event.type == "response.failed":
                raise RuntimeError("The model response failed.")

    except (OpenAIError, RuntimeError) as error:
        print(f"\nRequest failed: {error}")
        return None

    print("\nThe response ended unexpectedly.")
    return None


def print_history(history: list[Message]) -> None:
    for message in history:
        print(f"{message['role']}: {message['content']}")


def main() -> None:
    history: list[Message] = []

    print("AI Streaming Chatbot")
    print("Commands: /clear, /history, /exit")

    while True:
        prompt = input("\nYou: ").strip()

        if not prompt:
            continue

        command = prompt.lower()

        if command in EXIT_COMMANDS:
            print("Goodbye!")
            break

        if command in CLEAR_COMMANDS:
            history.clear()
            print("Conversation history cleared.")
            continue

        if command == "/history":
            print_history(history)
            continue

        recent_history = history[-MAX_HISTORY_MESSAGES:]
        messages = [*recent_history, {"role": "user", "content": prompt}]

        print("\nAI: ", end="", flush=True)
        reply = stream_reply(messages)

        # Save the turn only after a complete reply.
        if reply is not None:
            history.append({"role": "user", "content": prompt})
            history.append({"role": "assistant", "content": reply})


if __name__ == "__main__":
    main()
