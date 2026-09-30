import React from "react";
import { X, BarChart3, TrendingUp, Compass, Layers } from "lucide-react";

export default function AnalyticsPlotsModal({ onClose }) {
  const plots = [
    {
      title: "Yunnan Golden Benchmark (N=99) — 80.42% Mean / 84.76% Median Dice",
      src: "/api/plots/golden_dataset_benchmark_comparison.png",
      desc: "Cross-center zero-shot validation across Small (<10mL), Medium (10-25mL), and Large (>25mL) volume cohorts."
    },
    {
      title: "Multimodal Nuclear VLM Master Validation Dashboard",
      src: "/api/plots/master_nuclear_dashboard.png",
      desc: "Progressive loss convergence, confusion matrix, depth IoU regression, and tri-planar consensus metrics."
    },
    {
      title: "Tri-Planar Visual Hull Consensus Reconstruction",
      src: "/api/plots/triplanar_consensus_visual.png",
      desc: "Axial, Coronal, and Sagittal orthographic back-projection maintaining high-frequency boundary fidelity."
    },
    {
      title: "Longitudinal Depth Interval Regression (Z-Span)",
      src: "/api/plots/depth_interval_regression.png",
      desc: "Predicted slice bounds vs Expert Ground Truth slice indices, eliminating out-of-slice false positives."
    }
  ];

  return (
    <div className="pro-glass-panel floating-plots-modal">
      <div style={{ display: "flex", justifyContent: "space-between", alignItems: "center", marginBottom: "14px" }}>
        <div style={{ display: "flex", alignItems: "center", gap: "8px" }}>
          <BarChart3 size={18} color="var(--cyan)" />
          <span style={{ fontSize: "14px", fontWeight: 800, color: "#fff" }}>
            Benchmark Analytics & Empirical Plots
          </span>
        </div>
        <button className="btn-icon" onClick={onClose}>
          <X size={15} />
        </button>
      </div>

      <div style={{ display: "flex", flexDirection: "column", gap: "20px" }}>
        {plots.map((p, idx) => (
          <div key={idx} style={{ background: "rgba(6, 10, 18, 0.7)", borderRadius: "8px", padding: "12px", border: "1px solid var(--border-hairline)" }}>
            <div style={{ fontWeight: 700, fontSize: "12px", color: "var(--cyan)", marginBottom: "4px" }}>
              {p.title}
            </div>
            <p style={{ fontSize: "11px", color: "var(--text-dim)", marginBottom: "8px" }}>
              {p.desc}
            </p>
            <img
              src={p.src}
              alt={p.title}
              style={{ width: "100%", height: "auto", borderRadius: "6px", display: "block" }}
              loading="lazy"
            />
          </div>
        ))}
      </div>
    </div>
  );
}
