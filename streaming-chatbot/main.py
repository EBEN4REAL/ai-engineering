
import os

from openai import OpenAI, OpenAIError

client = OpenAI()

MODEL = os.getenv("OPENAI_MODEL", "gpt-4.1-mini")

SYSTEM_INSTRUCTIONS = """
You are a helpful AI assistant.

Be accurate, clear, and concise.
specialise in explaining software engineering and AI concepts.
If you don't know something, acknowledge the uncertainty.
"""

MAX_HISTORY_MESSAGES = 12

conversation_history = []


def chat(prompt: str) -> None:
    global conversation_history

    # Keep the most recent conversation turns.
    recent_history = conversation_history[-MAX_HISTORY_MESSAGES:]

    # Build the input without modifying saved history yet.
    messages = [
        *recent_history,
        {"role": "user", "content": prompt},
    ]

    response_text = ""
    completed = False

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
                response_text += event.delta

            elif event.type == "response.completed":
                completed = True

            elif event.type == "response.failed":
                raise RuntimeError("The model response failed.")

        print()

        if not completed:
            print("The response ended unexpectedly.")
            return

        # Save both messages only after successful completion.
        conversation_history = [
            *recent_history,
            {"role": "user", "content": prompt},
            {"role": "assistant", "content": response_text},
        ]

    except (OpenAIError, RuntimeError) as error:
        print(f"\nRequest failed: {error}")


def main() -> None:
    global conversation_history

    print("AI Streaming Chatbot")
    print("Commands: /clear, /history, /exit, /reset")

    while True:
        prompt = input("\nYou: ").strip()

        if not prompt:
            continue

        if prompt.lower() in {"/exit", "exit", "quit"}:
            print("Goodbye!")
            break

        if prompt.lower() == "/clear" or prompt.lower() == "/reset":
            conversation_history = []
            print("Conversation history cleared.")
            continue

        if prompt.lower() == "/history":
            for message in conversation_history:
                print(
                    f"{message['role']}: "
                    f"{message['content']}"
                )
            continue

        print("\nAI: ", end="", flush=True)
        chat(prompt)


if __name__ == "__main__": # if the main.py file is being run directly (not imported as a module), then execute the main function
    main()
