// Three dots pulsing in a regular wave — the AI-thinking indicator (问题4).
// The animation keyframes live in globals.css (.thinking-dots).

export function ThinkingDots() {
  return (
    <span className="thinking-dots" role="status" aria-label="AI 正在思考">
      <span />
      <span />
      <span />
    </span>
  );
}
