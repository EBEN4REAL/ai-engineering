import {
  useEffect,
  useRef,
  useState,
  type FormEvent,
  type KeyboardEvent,
} from "react";

// Keep in sync with the backend's MAX_MESSAGE_LENGTH.
const MAX_MESSAGE_LENGTH = 10_000;

type ChatComposerProps = {
  isStreaming: boolean;
  onSend: (text: string) => void;
  onStop: () => void;
};

function SendIcon() {
  return (
    <svg width="18" height="18" viewBox="0 0 24 24" fill="none" aria-hidden="true">
      <path
        d="M12 19V5m0 0-6 6m6-6 6 6"
        stroke="currentColor"
        strokeWidth="2.2"
        strokeLinecap="round"
        strokeLinejoin="round"
      />
    </svg>
  );
}

function StopIcon() {
  return (
    <svg width="14" height="14" viewBox="0 0 24 24" aria-hidden="true">
      <rect x="4" y="4" width="16" height="16" rx="3" fill="currentColor" />
    </svg>
  );
}

export function ChatComposer({ isStreaming, onSend, onStop }: ChatComposerProps) {
  const [value, setValue] = useState("");
  const textareaRef = useRef<HTMLTextAreaElement>(null);

  const canSend = value.trim().length > 0 && !isStreaming;

  // Grow the textarea with its content, up to the CSS max-height.
  useEffect(() => {
    const textarea = textareaRef.current;
    if (!textarea) return;

    textarea.style.height = "auto";
    textarea.style.height = `${textarea.scrollHeight}px`;
  }, [value]);

  function submit() {
    if (!canSend) return;

    onSend(value);
    setValue("");
  }

  function handleSubmit(event: FormEvent<HTMLFormElement>) {
    event.preventDefault();
    submit();
  }

  function handleKeyDown(event: KeyboardEvent<HTMLTextAreaElement>) {
    // Enter sends; Shift+Enter inserts a newline. Ignore Enter during IME composition.
    if (event.key === "Enter" && !event.shiftKey && !event.nativeEvent.isComposing) {
      event.preventDefault();
      submit();
    }
  }

  return (
    <div className="composer-area">
      <form className="composer" onSubmit={handleSubmit}>
        <textarea
          ref={textareaRef}
          className="composer-input"
          value={value}
          onChange={(event) => setValue(event.target.value)}
          onKeyDown={handleKeyDown}
          placeholder="Ask about AI engineering…"
          aria-label="Message"
          rows={1}
          maxLength={MAX_MESSAGE_LENGTH}
        />

        {isStreaming ? (
          <button
            type="button"
            className="send-button send-button--stop"
            onClick={onStop}
            aria-label="Stop generating"
          >
            <StopIcon />
          </button>
        ) : (
          <button
            type="submit"
            className="send-button"
            disabled={!canSend}
            aria-label="Send message"
          >
            <SendIcon />
          </button>
        )}
      </form>

      <p className="composer-hint">
        Enter to send · Shift + Enter for a new line
      </p>
    </div>
  );
}
