import React, { useState, useEffect } from "react";
import { api } from "../api";
import { Eye, EyeOff, Layers } from "lucide-react";

export default function TriPlanarViewer({ jobId, meta, lut = "default" }) {
  const [triplanar, setTriplanar] = useState(null);
  const [showMask, setShowMask] = useState(true);
  const [alpha, setAlpha] = useState(0.45);

  useEffect(() => {
    if (!jobId) return;
    api.triplanar(jobId)
      .then(setTriplanar)
      .catch(console.error);
  }, [jobId]);

  const axialIdx = triplanar?.views?.axial?.slice_idx ?? meta?.best_slices?.axial ?? 60;
  const coronalIdx = triplanar?.views?.coronal?.slice_idx ?? meta?.best_slices?.coronal ?? 192;
  const sagittalIdx = triplanar?.views?.sagittal?.slice_idx ?? meta?.best_slices?.sagittal ?? 192;

  const views = [
    { id: "axial", name: "Axial View (Z-plane)", slice_idx: axialIdx, max_slices: meta?.shape_xyz?.[2] || 120 },
    { id: "coronal", name: "Coronal View (Y-plane)", slice_idx: coronalIdx, max_slices: meta?.shape_xyz?.[1] || 384 },
    { id: "sagittal", name: "Sagittal View (X-plane)", slice_idx: sagittalIdx, max_slices: meta?.shape_xyz?.[0] || 384 },
  ];

  return (
    <div style={{ width: "100%", height: "100%", padding: "70px 20px 80px 20px", display: "flex", flexDirection: "column", gap: "12px", boxSizing: "border-box" }}>
      <div style={{ display: "flex", justifyContent: "space-between", alignItems: "center" }}>
        <div style={{ display: "flex", alignItems: "center", gap: "8px" }}>
          <Layers size={16} color="var(--cyan)" />
          <span style={{ fontWeight: 700, fontSize: "13px", color: "#fff" }}>Synchronized Tri-Planar Consensus</span>
        </div>
        <div style={{ display: "flex", alignItems: "center", gap: "10px" }}>
          <button
            className={`btn-cad ${showMask ? "active" : ""}`}
            onClick={() => setShowMask(!showMask)}
          >
            {showMask ? <Eye size={13} /> : <EyeOff size={13} />}
            <span>Overlay Mask</span>
          </button>
        </div>
      </div>

      <div style={{ display: "grid", gridTemplateColumns: "1fr 1fr 1fr", gap: "16px", flex: 1, minHeight: 0 }}>
        {views.map((v) => {
          const sliceUrl = api.sliceUrl(jobId, {
            plane: v.id,
            index: v.slice_idx,
            overlay: showMask ? 1 : 0,
            alpha,
            lut
          });

          return (
            <div
              key={v.id}
              className="pro-glass-panel ortho-viewport"
              style={{ position: "relative", display: "flex", alignItems: "center", justifyContent: "center", overflow: "hidden", background: "#020408" }}
            >
              <div style={{ position: "absolute", top: "10px", left: "10px", background: "rgba(6, 10, 18, 0.8)", padding: "4px 8px", borderRadius: "6px", border: "1px solid var(--border-hairline)", fontFamily: "var(--font-mono)", fontSize: "11px", color: "var(--cyan)", zIndex: 10 }}>
                {v.name} • Slice {v.slice_idx + 1}/{v.max_slices}
              </div>
              <img
                src={sliceUrl}
                alt={v.name}
                style={{ width: "100%", height: "100%", objectFit: "contain", borderRadius: "8px" }}
              />
            </div>
          );
        })}
      </div>
    </div>
  );
}
