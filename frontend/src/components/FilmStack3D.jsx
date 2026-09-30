import React, { useState, useEffect, useRef } from "react";
import { api } from "../api";
import { Play, Pause, RotateCw, Layers, Compass, Eye } from "lucide-react";

export default function FilmStack3D({ jobId, meta, lut = "default" }) {
  const [sliceCount, setSliceCount] = useState(meta?.shape_xyz?.[2] || 60);
  const [currentZ, setCurrentZ] = useState(meta?.best_slices?.axial || 30);
  const [spacing, setSpacing] = useState(14); // px between slices
  const [sliceOpacity, setSliceOpacity] = useState(0.55);
  const [orbit, setOrbit] = useState({ rotX: 55, rotZ: -32, zoom: 1.05 });
  const [isDragging, setIsDragging] = useState(false);
  const [dragStart, setDragStart] = useState({ x: 0, y: 0 });

  // Auto-sweep cine
  const [isSweeping, setIsSweeping] = useState(false);
  const [gtSlices, setGtSlices] = useState([]);

  // Fetch volume stack metadata
  useEffect(() => {
    if (!jobId) return;
    api.volumeStack(jobId)
      .then((data) => {
        if (data.slice_count) setSliceCount(data.slice_count);
        if (data.key_slice !== undefined) setCurrentZ(data.key_slice);
        if (data.gt_slices) setGtSlices(data.gt_slices);
      })
      .catch(console.error);
  }, [jobId]);

  // Orbit drag handlers
  const handleMouseDown = (e) => {
    setIsDragging(true);
    setDragStart({ x: e.clientX, y: e.clientY });
  };

  const handleMouseMove = (e) => {
    if (!isDragging) return;
    const dx = e.clientX - dragStart.x;
    const dy = e.clientY - dragStart.y;
    setDragStart({ x: e.clientX, y: e.clientY });

    setOrbit((prev) => ({
      rotZ: (prev.rotZ + dx * 0.4) % 360,
      rotX: Math.max(15, Math.min(88, prev.rotX - dy * 0.4)),
      zoom: prev.zoom
    }));
  };

  const handleMouseUp = () => setIsDragging(false);

  // Wheel zoom
  const handleWheel = (e) => {
    e.preventDefault();
    const delta = e.deltaY < 0 ? 0.08 : -0.08;
    setOrbit((prev) => ({
      ...prev,
      zoom: Math.max(0.6, Math.min(2.0, prev.zoom + delta))
    }));
  };

  // Auto-sweep effect
  useEffect(() => {
    if (!isSweeping) return;
    const t = setInterval(() => {
      setCurrentZ((prev) => (prev + 1) % sliceCount);
    }, 200);
    return () => clearInterval(t);
  }, [isSweeping, sliceCount]);

  // Presets
  const setPreset = (name) => {
    if (name === "iso") {
      setOrbit({ rotX: 52, rotZ: -30, zoom: 1.0 });
      setSpacing(14);
    } else if (name === "lateral") {
      setOrbit({ rotX: 75, rotZ: -20, zoom: 0.95 });
      setSpacing(24);
    } else if (name === "top") {
      setOrbit({ rotX: 88, rotZ: 0, zoom: 1.1 });
      setSpacing(6);
    }
  };

  // Render subset of representative slices (e.g., 20 translucent layers centered around currentZ)
  const layerStep = Math.max(1, Math.floor(sliceCount / 24));
  const renderedIndices = [];
  for (let z = 0; z < sliceCount; z += layerStep) {
    renderedIndices.push(z);
  }
  // Ensure currentZ is included in the rendered stack
  if (!renderedIndices.includes(currentZ)) {
    renderedIndices.push(currentZ);
    renderedIndices.sort((a, b) => a - b);
  }

  return (
    <div
      className="stack3d-container"
      onMouseDown={handleMouseDown}
      onMouseMove={handleMouseMove}
      onMouseUp={handleMouseUp}
      onMouseLeave={handleMouseUp}
      onWheel={handleWheel}
    >
      {/* 3D Viewport Controls HUD */}
      <div className="stack-hud-controls">
        <div style={{ display: "flex", justifyContent: "space-between", alignItems: "center" }}>
          <span style={{ fontSize: "11px", fontWeight: "700", color: "#fff", display: "flex", alignItems: "center", gap: "6px" }}>
            <Layers size={14} color="#00f0ff" /> 3D Film Stacker
          </span>
          <span style={{ fontFamily: "var(--font-mono)", fontSize: "10px", color: "var(--cyan)" }}>
            Z: {currentZ + 1}/{sliceCount}
          </span>
        </div>

        <div style={{ display: "flex", gap: "4px" }}>
          <button className="btn-cad" style={{ flex: 1 }} onClick={() => setPreset("iso")}>Isometric</button>
          <button className="btn-cad" style={{ flex: 1 }} onClick={() => setPreset("lateral")}>Exploded</button>
          <button className="btn-cad" style={{ flex: 1 }} onClick={() => setPreset("top")}>Planar</button>
        </div>

        <div>
          <div style={{ display: "flex", justifyContent: "space-between", fontSize: "10px", color: "var(--text-dim)", marginBottom: "3px" }}>
            <span>Layer Spacing</span>
            <span style={{ fontFamily: "var(--font-mono)", color: "var(--cyan)" }}>{spacing}px</span>
          </div>
          <input
            type="range"
            min="4"
            max="32"
            value={spacing}
            onChange={(e) => setSpacing(Number(e.target.value))}
            className="pacs-range-input"
            style={{ width: "100%" }}
          />
        </div>

        <div>
          <div style={{ display: "flex", justifyContent: "space-between", fontSize: "10px", color: "var(--text-dim)", marginBottom: "3px" }}>
            <span>Layer Opacity</span>
            <span style={{ fontFamily: "var(--font-mono)", color: "var(--cyan)" }}>{Math.round(sliceOpacity * 100)}%</span>
          </div>
          <input
            type="range"
            min="0.1"
            max="0.95"
            step="0.05"
            value={sliceOpacity}
            onChange={(e) => setSliceOpacity(Number(e.target.value))}
            className="pacs-range-input"
            style={{ width: "100%" }}
          />
        </div>

        <button
          className={`btn-cad ${isSweeping ? "active" : ""}`}
          style={{ width: "100%", marginTop: "4px" }}
          onClick={() => setIsSweeping(!isSweeping)}
        >
          {isSweeping ? <Pause size={13} /> : <Play size={13} />}
          <span>{isSweeping ? "Stop Volumetric Sweep" : "Start Volumetric Sweep"}</span>
        </button>
      </div>

      {/* 3D World */}
      <div
        className="stack3d-world"
        style={{
          transform: `rotateX(${orbit.rotX}deg) rotateZ(${orbit.rotZ}deg) scale(${orbit.zoom})`
        }}
      >
        {renderedIndices.map((z) => {
          const isActive = z === currentZ;
          const isGt = gtSlices.includes(z);
          // Elevation offset if active
          const elevation = isActive ? 28 : 0;
          const tz = (z - sliceCount / 2) * (spacing * 0.7) + elevation;
          const opacity = isActive ? 0.95 : sliceOpacity;

          const sliceUrl = api.sliceUrl(jobId, {
            plane: "axial",
            index: z,
            overlay: isGt || isActive ? 1 : 0,
            alpha: 0.55,
            lut
          });

          return (
            <div
              key={z}
              className={`film-slice-layer ${isActive ? "active-slice" : ""}`}
              style={{
                transform: `translateZ(${tz}px)`,
                opacity,
                filter: isGt ? "drop-shadow(0 0 8px rgba(16, 185, 129, 0.4))" : undefined
              }}
            >
              <img src={sliceUrl} alt={`Slice ${z}`} loading="lazy" />
            </div>
          );
        })}
      </div>
    </div>
  );
}
