import React, { useState } from "react";
import { Send, Sparkles, MessageSquare, Terminal, RefreshCw } from "lucide-react";
import { api } from "../api";

export default function VlmPromptBar({ jobId, modelType, onPromptResult }) {
  const [promptText, setPromptText] = useState("");
  const [isThinking, setIsThinking] = useState(false);

  const suggestions = [
    "Assess dynamic enhancement and wash-in kinetics",
    "Analyze lesion morphology and predict BI-RADS",
    "Verify craniocaudal depth span boundaries",
    "Examine kinetic curve type and malignancy risk",
  ];

  const handleSubmit = async (textToSend) => {
    const text = (textToSend || promptText).trim();
    if (!text || !jobId) return;
    setIsThinking(true);
    try {
      const res = await api.prompt(jobId, text, modelType);
      if (onPromptResult) onPromptResult(res);
      setPromptText("");
    } catch (err) {
      console.error("Prompt failed:", err);
    } finally {
      setIsThinking(false);
    }
  };

  return (
    <div className="vlm-prompt-box">
      <div style={{ display: "flex", justifyContent: "space-between", alignItems: "center" }}>
        <span style={{ fontSize: "10px", fontWeight: 700, textTransform: "uppercase", color: "var(--cyan)", display: "flex", alignItems: "center", gap: "4px" }}>
          <Sparkles size={12} /> Prompt Clinical VLM
        </span>
        {isThinking && (
          <span style={{ fontSize: "10px", color: "var(--amber)", display: "flex", alignItems: "center", gap: "4px" }}>
            <RefreshCw size={10} className="animate-spin" /> VLM Thinking...
          </span>
        )}
      </div>

      {/* Input row */}
      <div className="vlm-prompt-input-row">
        <input
          type="text"
          value={promptText}
          onChange={(e) => setPromptText(e.target.value)}
          onKeyDown={(e) => { if (e.key === "Enter") handleSubmit(); }}
          placeholder="Ask clinical VLM (e.g. assess margins, wash-in, BI-RADS)..."
          className="vlm-prompt-input"
          disabled={isThinking}
        />
        <button
          className="btn-icon active"
          style={{ width: "28px", height: "28px" }}
          onClick={() => handleSubmit()}
          disabled={isThinking || !promptText.trim()}
        >
          <Send size={13} />
        </button>
      </div>

      {/* Suggestion Chips */}
      <div className="prompt-suggestions-row">
        {suggestions.map((s, idx) => (
          <span
            key={idx}
            className="prompt-chip"
            onClick={() => { setPromptText(s); handleSubmit(s); }}
          >
            {s}
          </span>
        ))}
      </div>
    </div>
  );
}
