import React from "react";
import { Activity, X, AlertTriangle, CheckCircle2 } from "lucide-react";

export default function KineticCurveCard({ curveData, onClose }) {
  if (!curveData) return null;

  const {
    phases = ["P1", "P2", "P3", "P4", "P5"],
    relative_enhancement_pct = [0, 85, 92, 78, 62],
    curve_type = "Type III (Washout)",
    clinical_interpretation = "Washout detected.",
    badge_color = "#ef4444",
    risk_level = "High Risk",
    coords = {}
  } = curveData;

  // Generate SVG polyline points
  const maxVal = Math.max(...relative_enhancement_pct, 100);
  const minVal = Math.min(...relative_enhancement_pct, 0);
  const width = 280;
  const height = 90;
  const padding = 15;

  const points = relative_enhancement_pct.map((val, idx) => {
    const x = padding + (idx / (relative_enhancement_pct.length - 1)) * (width - 2 * padding);
    const y = height - padding - ((val - minVal) / (maxVal - minVal + 1e-5)) * (height - 2 * padding);
    return `${x},${y}`;
  }).join(" ");

  return (
    <div className="kinetic-probe-card">
      <div style={{ display: "flex", justifyContent: "space-between", alignItems: "center", marginBottom: "8px" }}>
        <div style={{ display: "flex", alignItems: "center", gap: "6px" }}>
          <Activity size={15} color={badge_color} />
          <span style={{ fontWeight: 700, fontSize: "12px", color: "#fff" }}>Kinetic Washout Probe</span>
        </div>
        <button className="btn-icon" style={{ width: "22px", height: "22px" }} onClick={onClose}>
          <X size={13} />
        </button>
      </div>

      <div style={{ display: "flex", justifyContent: "space-between", alignItems: "center", marginBottom: "6px" }}>
        <span
          style={{
            fontSize: "10px",
            fontWeight: 800,
            padding: "2px 8px",
            borderRadius: "4px",
            backgroundColor: `${badge_color}22`,
            color: badge_color,
            border: `1px solid ${badge_color}66`
          }}
        >
          {curve_type}
        </span>
        <span style={{ fontFamily: "var(--font-mono)", fontSize: "10px", color: "var(--text-dim)" }}>
          Norm ({coords.x_norm?.toFixed(2)}, {coords.y_norm?.toFixed(2)})
        </span>
      </div>

      {/* Mini SVG Kinetic Graph */}
      <svg className="kinetic-curve-svg" viewBox={`0 0 ${width} ${height}`}>
        {/* Baseline grid line */}
        <line
          x1={padding}
          y1={height - padding}
          x2={width - padding}
          y2={height - padding}
          stroke="rgba(255, 255, 255, 0.15)"
          strokeWidth="1"
        />

        {/* Enhancement curve path */}
        <polyline
          fill="none"
          stroke={badge_color}
          strokeWidth="2.5"
          strokeLinecap="round"
          strokeLinejoin="round"
          points={points}
        />

        {/* Phase Markers */}
        {relative_enhancement_pct.map((val, idx) => {
          const x = padding + (idx / (relative_enhancement_pct.length - 1)) * (width - 2 * padding);
          const y = height - padding - ((val - minVal) / (maxVal - minVal + 1e-5)) * (height - 2 * padding);
          return (
            <g key={idx}>
              <circle cx={x} cy={y} r="3.5" fill={badge_color} />
              <text x={x} y={height - 2} fill="#94a3b8" fontSize="8" textAnchor="middle" fontFamily="monospace">
                {phases[idx]}
              </text>
            </g>
          );
        })}
      </svg>

      <p style={{ fontSize: "11px", color: "var(--text-muted)", lineHeight: 1.35, margin: 0 }}>
        {clinical_interpretation}
      </p>
    </div>
  );
}
