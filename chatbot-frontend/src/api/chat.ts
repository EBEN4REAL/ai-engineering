export type ChatRole = "user" | "assistant";

export type ChatMessage = {
  role: ChatRole;
  content: string;
};

type StreamEvent =
  | { type: "delta"; text: string }
  | { type: "done" }
  | { type: "error"; message: string };

type StreamOptions = {
  message: string;
  history: ChatMessage[];
  onDelta: (text: string) => void;
  signal?: AbortSignal;
};

/** Errors that are safe to show to the user. */
export class ChatApiError extends Error {}

const API_URL: string =
  import.meta.env.VITE_API_URL ?? "http://127.0.0.1:8000";

const DATA_PREFIX = "data:";

/** Extracts and parses the JSON payload of one SSE block. */
function parseEvent(block: string): StreamEvent | null {
  const data = block
    .split("\n")
    .filter((line) => line.startsWith(DATA_PREFIX))
    .map((line) => line.slice(DATA_PREFIX.length).trimStart())
    .join("\n");

  if (!data) return null;

  try {
    return JSON.parse(data) as StreamEvent;
  } catch {
    throw new ChatApiError("Received a malformed response from the server.");
  }
}

/**
 * Sends a message and streams the assistant reply.
 * Resolves once the server signals completion; rejects on errors or abort.
 */
export async function streamChat({
  message,
  history,
  onDelta,
  signal,
}: StreamOptions): Promise<void> {
  const response = await fetch(`${API_URL}/chat/stream`, {
    method: "POST",
    headers: {
      "Content-Type": "application/json",
      Accept: "text/event-stream",
    },
    body: JSON.stringify({ message, history }),
    signal,
  });

  if (!response.ok) {
    throw new ChatApiError(
      `The server responded with status ${response.status}.`,
    );
  }

  if (!response.body) {
    throw new ChatApiError("Streaming is not supported in this browser.");
  }

  const reader = response.body
    .pipeThrough(new TextDecoderStream())
    .getReader();

  // Network chunks don't align with SSE events, so keep the unparsed tail.
  let buffer = "";
  let completed = false;

  try {
    while (!completed) {
      const { value, done } = await reader.read();
      if (done) break;

      buffer += value;

      const blocks = buffer.split("\n\n");
      buffer = blocks.pop() ?? "";

      for (const block of blocks) {
        const event = parseEvent(block);
        if (!event) continue;

        if (event.type === "delta") {
          onDelta(event.text);
        } else if (event.type === "error") {
          throw new ChatApiError(event.message);
        } else {
          completed = true;
          break;
        }
      }
    }
  } finally {
    reader.releaseLock();
  }

  if (!completed) {
    throw new ChatApiError("The response ended unexpectedly.");
  }
}
