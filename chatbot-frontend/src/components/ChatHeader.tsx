type ChatHeaderProps = {
  canClear: boolean;
  onClear: () => void;
};

export function ChatHeader({ canClear, onClear }: ChatHeaderProps) {
  return (
    <header className="app-header">
      <div className="brand">
        <span className="brand-mark" aria-hidden="true">
          ✦
        </span>
        <div>
          <h1 className="brand-title">AI Engineering Assistant</h1>
          <p className="brand-subtitle">Streaming answers · FastAPI + OpenAI</p>
        </div>
      </div>

      <button
        type="button"
        className="ghost-button"
        onClick={onClear}
        disabled={!canClear}
      >
        New chat
      </button>
    </header>
  );
}
