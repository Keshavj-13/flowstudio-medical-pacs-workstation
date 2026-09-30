import React from "react";
import { Sparkles, Cpu, CheckCircle, ShieldCheck } from "lucide-react";

export default function ModelSelector({ selectedModel, onSelectModel, models }) {
  const defaultModels = [
    {
      id: "qwen_nuclear_vlm",
      name: "Qwen3.5-0.8B Nuclear Multimodal VLM",
      badge: "CHAMPION",
      dice: "85.21% (MamaMia) / 80.42% (Yunnan Golden)",
      desc: "Interwoven dual-stream foundation model with 16 spatial-kinetic tokens, tri-planar visual hull consensus, and clinical chain-of-thought diagnostics.",
      features: ["3D Voxel Segmentation", "Longitudinal Depth Bounding", "BI-RADS Classification", "Tri-Planar Consensus"]
    },
    {
      id: "flexible_unet",
      name: "FlexibleUNet-ResNet50 Baseline",
      badge: "LEGACY CNN",
      dice: "76.55%",
      desc: "Standard 3D sliding-window convolutional neural network without multimodal language cross-attention or kinetic reasoning.",
      features: ["Sliding Window 3D UNet", "Fixed Convolution"]
    }
  ];

  const modelList = (models && models.length > 0) ? models : defaultModels;

  return (
    <div className="card" style={{ marginBottom: "1.5rem" }}>
      <div className="cardHead">
        <h2 className="h2" style={{ display: "flex", alignItems: "center", gap: "8px" }}>
          <Sparkles size={20} color="var(--accent-cyan)" /> Model Architecture &amp; Reasoning Engine
        </h2>
        <div className="pill done" style={{ fontSize: "0.75rem" }}>
          Active: {selectedModel === "qwen_nuclear_vlm" ? "Qwen Nuclear VLM" : "FlexibleUNet"}
        </div>
      </div>

      <div style={{ display: "grid", gridTemplateColumns: "1fr 1fr", gap: "1rem", marginTop: "0.5rem" }}>
        {modelList.map((m) => {
          const isSelected = selectedModel === m.id;
          const isChampion = m.id === "qwen_nuclear_vlm";
          return (
            <div
              key={m.id}
              onClick={() => onSelectModel(m.id)}
              style={{
                border: isSelected
                  ? "2px solid var(--accent-cyan, #06b6d4)"
                  : "1px solid rgba(255, 255, 255, 0.12)",
                backgroundColor: isSelected ? "rgba(6, 182, 212, 0.08)" : "rgba(255, 255, 255, 0.03)",
                borderRadius: "10px",
                padding: "1rem",
                cursor: "pointer",
                transition: "all 0.2s ease-in-out",
                position: "relative"
              }}
            >
              <div style={{ display: "flex", justifyContent: "space-between", alignItems: "flex-start" }}>
                <div>
                  <div style={{ display: "flex", alignItems: "center", gap: "6px", marginBottom: "4px" }}>
                    <span style={{ fontWeight: 700, fontSize: "1rem", color: isSelected ? "#fff" : "#e2e8f0" }}>
                      {m.name}
                    </span>
                    {isChampion && (
                      <span
                        style={{
                          backgroundColor: "#16a34a",
                          color: "#fff",
                          fontSize: "0.65rem",
                          fontWeight: 700,
                          padding: "2px 6px",
                          borderRadius: "4px"
                        }}
                      >
                        {m.badge || "CHAMPION"}
                      </span>
                    )}
                  </div>
                  <div style={{ color: "#38bdf8", fontWeight: 600, fontSize: "0.85rem", marginBottom: "6px" }}>
                    Genuine 3D Dice: {m.dice || m.dice_score}
                  </div>
                </div>
                {isSelected && <CheckCircle size={20} color="var(--accent-cyan, #06b6d4)" />}
              </div>

              <div style={{ fontSize: "0.8rem", color: "#94a3b8", marginBottom: "8px", lineHeight: 1.4 }}>
                {m.desc || m.description}
              </div>

              <div style={{ display: "flex", flexWrap: "wrap", gap: "4px" }}>
                {(m.features || []).map((feat, idx) => (
                  <span
                    key={idx}
                    style={{
                      fontSize: "0.7rem",
                      backgroundColor: "rgba(255, 255, 255, 0.06)",
                      color: "#cbd5e1",
                      padding: "2px 6px",
                      borderRadius: "4px"
                    }}
                  >
                    • {feat}
                  </span>
                ))}
              </div>
            </div>
          );
        })}
      </div>
    </div>
  );
}
