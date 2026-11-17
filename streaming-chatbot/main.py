
from dotenv import load_dotenv
from openai import OpenAI

load_dotenv() 

client = OpenAI()

def chat(prompt: str) -> None:
    stream = client.responses.create(
        model="gpt-5.5",
        input=prompt,
        stream=True,
    )

    for event in stream:
        if event.type == "response.output_text.delta":
            print(event.delta, end="", flush=True)
  
    print()


if __name__ == "__main__": #run only when the script is executed directly, not when another file imports it
    while True:
        prompt = input("\nYou: ").strip()

        if prompt.lower() in {"exit", "quit"}:
            break

        if not prompt:
            continue

        print("AI: ", end="", flush=True)
        try:
            chat(prompt)
        except Exception as exc:
            print(f"\nRequest failed: {exc}")
