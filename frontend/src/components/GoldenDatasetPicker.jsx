import React, { useState } from "react";
import { Award, Play, Database, CheckCircle2, ChevronRight } from "lucide-react";

export default function GoldenDatasetPicker({ onSelectCase, loadingCaseId }) {
  const goldenCases = [
    {
      id: "yunnan_19",
      name: "Case #19 (Peak Volume)",
      pathology: "Malignant",
      volume_ml: 14.27,
      tier: "Medium",
      expected_dice: "96.42%",
      slices: 120,
      notes: "Peak volume segmentation across the entire external benchmark."
    },
    {
      id: "yunnan_9",
      name: "Case #9 (Solitary High-Washout)",
      pathology: "Malignant",
      volume_ml: 5.52,
      tier: "Small",
      expected_dice: "93.03%",
      slices: 120,
      notes: "Rapid type-III washout with high peak contrast enhancement."
    },
    {
      id: "yunnan_7",
      name: "Case #7 (Circumscribed IDC)",
      pathology: "Malignant",
      volume_ml: 9.05,
      tier: "Small",
      expected_dice: "91.35%",
      slices: 120,
      notes: "Well-circumscribed margin requiring precise longitudinal depth gating."
    },
    {
      id: "yunnan_1",
      name: "Case #1 (Benchmark Solitary)",
      pathology: "Malignant",
      volume_ml: 4.68,
      tier: "Small",
      expected_dice: "87.07%",
      slices: 120,
      notes: "Standard reference volume with ground-truth expert segmentation."
    },
    {
      id: "yunnan_2",
      name: "Case #2 (Large Enhancing Mass)",
      pathology: "Malignant",
      volume_ml: 37.49,
      tier: "Large",
      expected_dice: "87.13%",
      slices: 120,
      notes: "Multifocal 153k voxel volume with intense parenchymal background."
    },
    {
      id: "yunnan_5",
      name: "Case #5 (Lobulated Carcinoma)",
      pathology: "Malignant",
      volume_ml: 16.40,
      tier: "Medium",
      expected_dice: "88.01%",
      slices: 120,
      notes: "Spiculated lobulated margins validated with tri-planar consensus."
    }
  ];

  return (
    <div className="card" style={{ marginBottom: "1.5rem" }}>
      <div className="cardHead">
        <div>
          <h2 className="h2" style={{ display: "flex", alignItems: "center", gap: "8px" }}>
            <Award size={20} color="#eab308" /> Yunnan Golden Benchmark Test Suite
          </h2>
          <div className="muted small" style={{ marginTop: "2px" }}>
            99 3D DCE-MRI external cross-center volumes • Cohort Mean 3D Dice: <strong>80.42%</strong> (Target: &ge;80.0%)
          </div>
        </div>
        <div className="pill" style={{ borderColor: "#eab308", color: "#facc15" }}>
          External Golden Ground Truth
        </div>
      </div>

      <div
        style={{
          display: "grid",
          gridTemplateColumns: "repeat(auto-fit, minmax(260px, 1fr))",
          gap: "0.85rem",
          marginTop: "1rem"
        }}
      >
        {goldenCases.map((gc) => {
          const isLoading = loadingCaseId === gc.id;
          return (
            <div
              key={gc.id}
              style={{
                border: "1px solid rgba(255, 255, 255, 0.1)",
                backgroundColor: "rgba(255, 255, 255, 0.02)",
                borderRadius: "8px",
                padding: "0.85rem",
                display: "flex",
                flexDirection: "column",
                justifyContent: "space-between"
              }}
            >
              <div>
                <div style={{ display: "flex", justifyContent: "space-between", alignItems: "center", marginBottom: "4px" }}>
                  <span style={{ fontWeight: 600, fontSize: "0.95rem", color: "#f8fafc" }}>
                    {gc.name}
                  </span>
                  <span
                    style={{
                      backgroundColor: "#16a34a",
                      color: "#fff",
                      fontSize: "0.7rem",
                      fontWeight: 700,
                      padding: "2px 6px",
                      borderRadius: "4px"
                    }}
                  >
                    Dice: {gc.expected_dice}
                  </span>
                </div>

                <div style={{ display: "flex", gap: "6px", fontSize: "0.75rem", color: "#94a3b8", marginBottom: "6px" }}>
                  <span>Vol: {gc.volume_ml} mL ({gc.tier})</span>
                  <span>•</span>
                  <span>{gc.slices} slices</span>
                </div>

                <div style={{ fontSize: "0.78rem", color: "#cbd5e1", lineHeight: 1.35, marginBottom: "8px" }}>
                  {gc.notes}
                </div>
              </div>

              <button
                className="btn primary small"
                disabled={Boolean(loadingCaseId)}
                onClick={() => onSelectCase(gc.id)}
                style={{
                  width: "100%",
                  display: "flex",
                  alignItems: "center",
                  justifyContent: "center",
                  gap: "6px",
                  backgroundColor: isLoading ? "#475569" : "#0284c7"
                }}
              >
                {isLoading ? (
                  "Loading Case & Running VLM..."
                ) : (
                  <>
                    <Play size={14} /> Test Golden Case
                  </>
                )}
              </button>
            </div>
          );
        })}
      </div>
    </div>
  );
}
