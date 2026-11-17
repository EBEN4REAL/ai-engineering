import { ChatComposer } from "./components/ChatComposer";
import { ChatHeader } from "./components/ChatHeader";
import { MessageList } from "./components/MessageList";
import { useChat } from "./hooks/useChat";
import "./App.css";

export default function App() {
  const { messages, isStreaming, error, send, stop, clear } = useChat();

  return (
    <div className="app">
      <ChatHeader
        canClear={!isStreaming && messages.length > 0}
        onClear={clear}
      />

      <MessageList
        messages={messages}
        isStreaming={isStreaming}
        onSuggestionSelect={send}
      />

      {error && (
        <div className="error-banner" role="alert">
          {error}
        </div>
      )}

      <ChatComposer isStreaming={isStreaming} onSend={send} onStop={stop} />
    </div>
  );
}
