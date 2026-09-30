import { useEffect, useRef, useState, useCallback } from "react";
import { xaiApi } from "../xaiApi";
import { Box, RotateCcw, Eye } from "lucide-react";

/**
 * 3D Volume Renderer using WebGL Canvas.
 * 
 * Renders a rotatable 3D isosurface from the Grad-CAM or Occlusion
 * volume data using a lightweight ray-marching approach on 2D canvas.
 * 
 * For full VTK/Three.js integration, this component provides the
 * scaffolding — currently uses a maximum-intensity projection (MIP)
 * rendering which is lightweight and dependency-free.
 */
export default function XaiVolumeRenderer({ jobId, shapeXYZ }) {
  const canvasRef = useRef(null);
  const [volumeKind, setVolumeKind] = useState("gradcam");
  const [volumeData, setVolumeData] = useState(null);
  const [volumeShape, setVolumeShape] = useState(null);
  const [loading, setLoading] = useState(false);
  const [error, setError] = useState(null);
  const [rotation, setRotation] = useState({ rx: -25, ry: 35 });
  const [opacity, setOpacity] = useState(0.6);
  const [threshold, setThreshold] = useState(0.2);
  const dragging = useRef(false);
  const lastMouse = useRef({ x: 0, y: 0 });

  // Load volume data
  const loadVolume = useCallback(async () => {
    setLoading(true);
    setError(null);
    try {
      const { shape, data } = await xaiApi.volumeData(jobId, volumeKind);
      setVolumeShape(shape);
      setVolumeData(data);
    } catch (e) {
      setError(e.message);
    }
    setLoading(false);
  }, [jobId, volumeKind]);

  useEffect(() => { loadVolume(); }, [loadVolume]);

  // Render MIP projection
  useEffect(() => {
    if (!canvasRef.current || !volumeData || !volumeShape || volumeShape.length < 3) return;

    const canvas = canvasRef.current;
    const ctx = canvas.getContext("2d");
    const W = canvas.width;
    const H = canvas.height;

    const [D, VH, VW] = volumeShape;
    const rx = (rotation.rx * Math.PI) / 180;
    const ry = (rotation.ry * Math.PI) / 180;

    // Clear
    ctx.fillStyle = "#0a0a1a";
    ctx.fillRect(0, 0, W, H);

    // Simple maximum intensity projection with rotation
    const cosRx = Math.cos(rx), sinRx = Math.sin(rx);
    const cosRy = Math.cos(ry), sinRy = Math.sin(ry);

    const maxDim = Math.max(D, VH, VW);
    const scale = Math.min(W, H) / (maxDim * 1.5);
    const cx = W / 2, cy = H / 2;

    const imageData = ctx.createImageData(W, H);
    const pixels = imageData.data;

    // Sample along rays through the volume
    const steps = Math.ceil(maxDim * 1.4);
    const halfD = D / 2, halfH = VH / 2, halfW = VW / 2;

    for (let py = 0; py < H; py += 2) {
      for (let px = 0; px < W; px += 2) {
        let maxVal = 0;

        const screenX = (px - cx) / scale;
        const screenY = (py - cy) / scale;

        for (let s = -steps / 2; s < steps / 2; s++) {
          // Ray direction (into screen, rotated)
          let x = screenX;
          let y = screenY;
          let z = s;

          // Rotate around Y
          const x1 = x * cosRy - z * sinRy;
          const z1 = x * sinRy + z * cosRy;
          // Rotate around X
          const y1 = y * cosRx - z1 * sinRx;
          const z2 = y * sinRx + z1 * cosRx;

          // Map to volume coordinates
          const vi = Math.round(z2 + halfD);
          const vj = Math.round(y1 + halfH);
          const vk = Math.round(x1 + halfW);

          if (vi >= 0 && vi < D && vj >= 0 && vj < VH && vk >= 0 && vk < VW) {
            const val = volumeData[vi * VH * VW + vj * VW + vk];
            if (val > threshold) {
              maxVal = Math.max(maxVal, val);
            }
          }
        }

        // Color mapping (clinical: blue → cyan → yellow)
        const v = maxVal * opacity;
        let r, g, b;
        if (volumeKind === "gradcam") {
          r = Math.min(255, Math.floor(v * 2 * 255));
          g = Math.min(255, Math.floor(v * 1.5 * 200));
          b = Math.min(255, Math.floor((1 - v) * 200 + 55));
        } else {
          r = Math.min(255, Math.floor(v * 2.5 * 255));
          g = Math.min(255, Math.floor(v * 1.2 * 180));
          b = Math.min(255, Math.floor(v * 0.3 * 80));
        }

        const a = v > 0.01 ? 255 : 0;

        // Write 2×2 block for performance
        for (let dy = 0; dy < 2 && py + dy < H; dy++) {
          for (let dx = 0; dx < 2 && px + dx < W; dx++) {
            const idx = ((py + dy) * W + (px + dx)) * 4;
            pixels[idx] = r;
            pixels[idx + 1] = g;
            pixels[idx + 2] = b;
            pixels[idx + 3] = a;
          }
        }
      }
    }

    ctx.putImageData(imageData, 0, 0);

    // Labels
    ctx.fillStyle = "rgba(148, 163, 184, 0.8)";
    ctx.font = "11px Inter, sans-serif";
    ctx.fillText(`${volumeKind.toUpperCase()} 3D MIP`, 10, 20);
    ctx.fillText(`Volume: ${D}×${VH}×${VW}`, 10, 36);
    ctx.fillText(`Rotation: ${rotation.rx.toFixed(0)}°, ${rotation.ry.toFixed(0)}°`, 10, 52);

  }, [volumeData, volumeShape, rotation, opacity, threshold, volumeKind]);

  // Mouse rotation
  const handleMouseDown = (e) => {
    dragging.current = true;
    lastMouse.current = { x: e.clientX, y: e.clientY };
  };

  const handleMouseMove = (e) => {
    if (!dragging.current) return;
    const dx = e.clientX - lastMouse.current.x;
    const dy = e.clientY - lastMouse.current.y;
    setRotation(prev => ({
      rx: prev.rx + dy * 0.5,
      ry: prev.ry + dx * 0.5,
    }));
    lastMouse.current = { x: e.clientX, y: e.clientY };
  };

  const handleMouseUp = () => { dragging.current = false; };

  return (
    <div className="xai-3d-container">
      <div className="xai-3d-header">
        <div className="xai-3d-title">
          <Box size={18} /> 3D Volume Rendering
        </div>
        <div className="xai-3d-controls">
          <select className="xai-select" value={volumeKind}
            onChange={e => setVolumeKind(e.target.value)}>
            <option value="gradcam">Grad-CAM</option>
            <option value="occlusion">Occlusion</option>
          </select>
          <label className="xai-toolbar-label">
            <Eye size={12} />
            <input type="range" min="0.1" max="1" step="0.05" value={opacity}
              onChange={e => setOpacity(Number(e.target.value))} />
          </label>
          <label className="xai-toolbar-label">
            Threshold
            <input type="range" min="0" max="0.8" step="0.05" value={threshold}
              onChange={e => setThreshold(Number(e.target.value))} />
          </label>
          <button className="xai-nav-btn" onClick={() => setRotation({ rx: -25, ry: 35 })}>
            <RotateCcw size={14} />
          </button>
        </div>
      </div>

      <div className="xai-3d-canvas-wrapper">
        {loading && <div className="xai-3d-loading">Loading volume data...</div>}
        {error && <div className="xai-3d-error">{error}</div>}
        <canvas
          ref={canvasRef}
          width={400}
          height={400}
          className="xai-3d-canvas"
          onMouseDown={handleMouseDown}
          onMouseMove={handleMouseMove}
          onMouseUp={handleMouseUp}
          onMouseLeave={handleMouseUp}
        />
      </div>
    </div>
  );
}
