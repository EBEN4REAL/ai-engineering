import { useCallback, useEffect, useRef, useState } from "react";
import {
  ChatApiError,
  streamChat,
  type ChatMessage,
  type TokenUsage,
} from "../api/chat";

export type Message = ChatMessage & {
  id: string;
  usage?: TokenUsage | null;
  stopped?: boolean;
};

type CommitOptions = {
  usage?: TokenUsage | null;
  stopped?: boolean;
};

type PendingTurn = {
  userText: string;
  reply: string;
};

// Keep in sync with the backend's MAX_HISTORY_MESSAGES.
const MAX_HISTORY_MESSAGES = 12;
const GENERIC_ERROR = "Something went wrong. Please try again.";

export function useChat() {
  const [history, setHistory] = useState<Message[]>([]);
  const [pending, setPending] = useState<PendingTurn | null>(null);
  const [error, setError] = useState<string | null>(null);
  const abortRef = useRef<AbortController | null>(null);

  const isStreaming = pending !== null;

  const send = useCallback(
    async (text: string) => {
      const message = text.trim();
      if (!message || abortRef.current) return;

      const controller = new AbortController();
      abortRef.current = controller;

      const apiHistory: ChatMessage[] = history
        .slice(-MAX_HISTORY_MESSAGES)
        .map(({ role, content }) => ({ role, content }));

      const commitTurn = (reply: string, options: CommitOptions = {}) => {
        setHistory((prev) => [
          ...prev,
          { id: crypto.randomUUID(), role: "user", content: message },
          {
            id: crypto.randomUUID(),
            role: "assistant",
            content: reply,
            usage: options.usage ?? null,
            stopped: options.stopped ?? false,
          },
        ]);
      };

      let reply = "";
      setError(null);
      setPending({ userText: message, reply });

      try {
        const usage = await streamChat({
          message,
          history: apiHistory,
          signal: controller.signal,
          onDelta: (delta) => {
            reply += delta;
            setPending({ userText: message, reply });
          },
        });

        commitTurn(reply, { usage });
      } catch (err) {
        if (controller.signal.aborted) {
          // Keep the partial reply and mark it, so the user knows it's incomplete.
          // The user's message is kept even if no text had arrived yet.
          commitTurn(reply, { stopped: true });
        } else {
          setError(err instanceof ChatApiError ? err.message : GENERIC_ERROR);
        }
      } finally {
        abortRef.current = null;
        setPending(null);
      }
    },
    [history],
  );

  const stop = useCallback(() => {
    abortRef.current?.abort();
  }, []);

  const clear = useCallback(() => {
    if (abortRef.current) return;
    setHistory([]);
    setError(null);
  }, []);

  // Cancel any in-flight request when the component unmounts.
  useEffect(() => () => abortRef.current?.abort(), []);

  const messages: Message[] = pending
    ? [
        ...history,
        { id: "pending-user", role: "user", content: pending.userText },
        { id: "pending-assistant", role: "assistant", content: pending.reply },
      ]
    : history;

  return { messages, isStreaming, error, send, stop, clear };
}
