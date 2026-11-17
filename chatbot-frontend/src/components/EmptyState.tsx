const SUGGESTIONS = [
  "Explain retrieval-augmented generation (RAG) with a practical example",
  "What's the difference between fine-tuning and prompt engineering?",
  "How do I evaluate an LLM application before shipping it?",
  "Design a streaming chat API using Server-Sent Events",
];

type EmptyStateProps = {
  onSelect: (prompt: string) => void;
};

export function EmptyState({ onSelect }: EmptyStateProps) {
  return (
    <div className="empty-state">
      <h2 className="empty-title">What would you like to build?</h2>
      <p className="empty-subtitle">
        Ask about AI engineering, or start with one of these.
      </p>

      <div className="suggestions">
        {SUGGESTIONS.map((prompt) => (
          <button
            key={prompt}
            type="button"
            className="suggestion"
            onClick={() => onSelect(prompt)}
          >
            {prompt}
          </button>
        ))}
      </div>
    </div>
  );
}
