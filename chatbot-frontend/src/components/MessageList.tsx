import { useEffect, useRef } from "react";
import type { Message } from "../hooks/useChat";
import { EmptyState } from "./EmptyState";
import { MessageBubble } from "./MessageBubble";

type MessageListProps = {
  messages: Message[];
  isStreaming: boolean;
  onSuggestionSelect: (prompt: string) => void;
};

export function MessageList({
  messages,
  isStreaming,
  onSuggestionSelect,
}: MessageListProps) {
  const endRef = useRef<HTMLDivElement>(null);

  // Keep the latest content in view. Skip smooth scrolling while tokens stream
  // in, since it would lag behind the text.
  useEffect(() => {
    endRef.current?.scrollIntoView({
      block: "end",
      behavior: isStreaming ? "auto" : "smooth",
    });
  }, [messages, isStreaming]);

  return (
    <section className="chat-scroll" aria-label="Conversation">
      <div className="chat-column" role="log" aria-live="polite">
        {messages.length === 0 ? (
          <EmptyState onSelect={onSuggestionSelect} />
        ) : (
          messages.map((message) => (
            <MessageBubble key={message.id} message={message} />
          ))
        )}
        <div ref={endRef} />
      </div>
    </section>
  );
}
