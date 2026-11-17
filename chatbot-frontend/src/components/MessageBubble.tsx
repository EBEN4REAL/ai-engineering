import Markdown from "react-markdown";
import type { Message } from "../hooks/useChat";

type MessageBubbleProps = {
  message: Message;
};

function TypingIndicator() {
  return (
    <span className="typing" role="status" aria-label="Assistant is typing">
      <span />
      <span />
      <span />
    </span>
  );
}

export function MessageBubble({ message }: MessageBubbleProps) {
  if (message.role === "user") {
    return (
      <div className="message message--user">
        <div className="bubble bubble--user">{message.content}</div>
      </div>
    );
  }

  return (
    <div className="message message--assistant">
      <div className="avatar" aria-hidden="true">
        AI
      </div>
      <div className="message-body">
        {message.content && (
          <div className="markdown">
            <Markdown>{message.content}</Markdown>
          </div>
        )}

        {!message.content && !message.stopped && <TypingIndicator />}

        {message.stopped && (
          <p className="stopped-note" role="status">
            <span className="stopped-dot" aria-hidden="true" />
            Generation stopped
          </p>
        )}

        {message.usage && (
          <p className="usage">
            {message.usage.input_tokens} in · {message.usage.output_tokens} out
            · {message.usage.total_tokens} total tokens
          </p>
        )}
      </div>
    </div>
  );
}
