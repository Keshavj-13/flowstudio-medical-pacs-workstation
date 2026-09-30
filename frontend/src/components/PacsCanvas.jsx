import React, { useRef, useEffect, useState, useCallback } from "react";
import { api } from "../api";
import {
  Play, Pause, ChevronLeft, ChevronRight, RotateCcw,
  Sliders, Eye, EyeOff, Activity, Crosshair, ZoomIn, ZoomOut, Maximize2
} from "lucide-react";

export default function PacsCanvas({
  jobId,
  meta,
  activeTool = "pan",
  lut = "default",
  onProbeResult = null,
  activePlane = "axial",
  onRunScan = null,
  isLoading = false,
}) {
  const canvasRef = useRef(null);
  const containerRef = useRef(null);

  // Slice state
  const maxSlices = meta?.shape_xyz?.[2] || 120;
  const [sliceIndex, setSliceIndex] = useState(meta?.best_slices?.axial ?? Math.floor(maxSlices / 2));
  const [isPlaying, setIsPlaying] = useState(false);
  const [fps, setFps] = useState(12);

  // Viewport transformation
  const [scale, setScale] = useState(1.0);
  const [pan, setPan] = useState({ x: 0, y: 0 });
  const [isDragging, setIsDragging] = useState(false);
  const [dragStart, setDragStart] = useState({ x: 0, y: 0 });

  // Radiological Knobs
  const [windowCenter, setWindowCenter] = useState(null);
  const [windowWidth, setWindowWidth] = useState(null);
  const [showMask, setShowMask] = useState(true);
  const [maskAlpha, setMaskAlpha] = useState(0.5);
  const [threshold, setThreshold] = useState(0.5);
  const [contourOnly, setContourOnly] = useState(false);

  // Interactive Tools state
  const [crosshairPos, setCrosshairPos] = useState({ x: 0, y: 0, val: "-" });
  const [boxROI, setBoxROI] = useState(null);
  const [isDrawingBox, setIsDrawingBox] = useState(false);
  const [boxStart, setBoxStart] = useState({ x: 0, y: 0 });

  // Loaded image cache
  const [currentImg, setCurrentImg] = useState(null);
  const [isLoadingSlice, setIsLoadingSlice] = useState(false);

  // Sync sliceIndex if meta changes
  useEffect(() => {
    if (meta?.best_slices?.axial !== undefined) {
      setSliceIndex(meta.best_slices.axial);
    }
  }, [meta]);

  // Fetch current slice image
  useEffect(() => {
    if (!jobId) return;
    let isMounted = true;
    setIsLoadingSlice(true);

    const url = api.sliceUrl(jobId, {
      plane: activePlane,
      index: sliceIndex,
      overlay: showMask ? 1 : 0,
      alpha: maskAlpha,
      threshold,
      lut,
      window_center: windowCenter,
      window_width: windowWidth,
      contour: contourOnly ? 1 : 0
    });

    const img = new Image();
    img.crossOrigin = "anonymous";
    img.src = url;
    img.onload = () => {
      if (isMounted) {
        setCurrentImg(img);
        setIsLoadingSlice(false);
      }
    };
    img.onerror = () => {
      if (isMounted) setIsLoadingSlice(false);
    };

    return () => { isMounted = false; };
  }, [jobId, activePlane, sliceIndex, showMask, maskAlpha, threshold, lut, windowCenter, windowWidth, contourOnly]);

  // Cine loop player
  useEffect(() => {
    if (!isPlaying) return;
    const interval = setInterval(() => {
      setSliceIndex((prev) => (prev + 1) % maxSlices);
    }, 1000 / fps);
    return () => clearInterval(interval);
  }, [isPlaying, fps, maxSlices]);

  // Render to canvas
  const renderCanvas = useCallback(() => {
    const canvas = canvasRef.current;
    if (!canvas || !currentImg) return;
    const ctx = canvas.getContext("2d");

    // Clear background
    ctx.fillStyle = "#04060a";
    ctx.fillRect(0, 0, canvas.width, canvas.height);

    ctx.save();
    // Center & Apply Viewport transform
    ctx.translate(canvas.width / 2 + pan.x, canvas.height / 2 + pan.y);
    ctx.scale(scale, scale);

    const iw = currentImg.width || 384;
    const ih = currentImg.height || 384;
    ctx.drawImage(currentImg, -iw / 2, -ih / 2, iw, ih);

    // Draw Box ROI if present
    if (boxROI) {
      const bx = boxROI.x * iw - iw / 2;
      const by = boxROI.y * ih - ih / 2;
      const bw = boxROI.w * iw;
      const bh = boxROI.h * ih;

      ctx.strokeStyle = activeTool === "blackout" ? "#ef4444" : "#00f0ff";
      ctx.lineWidth = 2 / scale;
      ctx.setLineDash([4 / scale, 4 / scale]);
      ctx.strokeRect(bx, by, bw, bh);

      if (activeTool === "blackout") {
        ctx.fillStyle = "rgba(239, 68, 68, 0.25)";
        ctx.fillRect(bx, by, bw, bh);
      }
    }

    ctx.restore();
  }, [currentImg, pan, scale, boxROI, activeTool]);

  useEffect(() => {
    renderCanvas();
  }, [renderCanvas]);

  // Resize canvas to parent
  useEffect(() => {
    const handleResize = () => {
      if (containerRef.current && canvasRef.current) {
        canvasRef.current.width = containerRef.current.clientWidth;
        canvasRef.current.height = containerRef.current.clientHeight;
        renderCanvas();
      }
    };
    handleResize();
    window.addEventListener("resize", handleResize);
    return () => window.removeEventListener("resize", handleResize);
  }, [renderCanvas]);

  // Coordinate conversion helper
  const getCanvasCoords = (e) => {
    const rect = canvasRef.current.getBoundingClientRect();
    const x = e.clientX - rect.left;
    const y = e.clientY - rect.top;
    return { x, y };
  };

  // Pan & Tool Mouse Events
  const handleMouseDown = (e) => {
    const { x, y } = getCanvasCoords(e);

    if (activeTool === "pan") {
      setIsDragging(true);
      setDragStart({ x: x - pan.x, y: y - pan.y });
    } else if (activeTool === "box" || activeTool === "crop" || activeTool === "blackout") {
      setIsDrawingBox(true);
      setBoxStart({ x, y });
    } else if (activeTool === "probe" && jobId) {
      // Calculate normalized coords [0, 1] relative to image
      const canvas = canvasRef.current;
      const iw = currentImg?.width || 384;
      const ih = currentImg?.height || 384;
      const cx = canvas.width / 2 + pan.x;
      const cy = canvas.height / 2 + pan.y;

      const imgX = (x - cx) / scale + iw / 2;
      const imgY = (y - cy) / scale + ih / 2;

      const normX = Math.max(0, Math.min(1, imgX / iw));
      const normY = Math.max(0, Math.min(1, imgY / ih));

      api.kineticCurve(jobId, { x_norm: normX, y_norm: normY, z_index: sliceIndex })
        .then((curve) => {
          if (onProbeResult) onProbeResult(curve);
        })
        .catch(console.error);
    }
  };

  const handleMouseMove = (e) => {
    const { x, y } = getCanvasCoords(e);

    // Update crosshair HUD
    const canvas = canvasRef.current;
    if (canvas && currentImg) {
      const iw = currentImg.width || 384;
      const ih = currentImg.height || 384;
      const cx = canvas.width / 2 + pan.x;
      const cy = canvas.height / 2 + pan.y;
      const imgX = Math.round((x - cx) / scale + iw / 2);
      const imgY = Math.round((y - cy) / scale + ih / 2);
      if (imgX >= 0 && imgX < iw && imgY >= 0 && imgY < ih) {
        setCrosshairPos({ x: imgX, y: imgY, val: `Slice ${sliceIndex + 1}` });
      }
    }

    if (isDragging && activeTool === "pan") {
      setPan({ x: x - dragStart.x, y: y - dragStart.y });
    } else if (isDrawingBox && (activeTool === "box" || activeTool === "crop" || activeTool === "blackout")) {
      const canvas = canvasRef.current;
      const iw = currentImg?.width || 384;
      const ih = currentImg?.height || 384;
      const cx = canvas.width / 2 + pan.x;
      const cy = canvas.height / 2 + pan.y;

      const startImgX = (boxStart.x - cx) / scale + iw / 2;
      const startImgY = (boxStart.y - cy) / scale + ih / 2;
      const curImgX = (x - cx) / scale + iw / 2;
      const curImgY = (y - cy) / scale + ih / 2;

      const minX = Math.min(startImgX, curImgX) / iw;
      const minY = Math.min(startImgY, curImgY) / ih;
      const w = Math.abs(curImgX - startImgX) / iw;
      const h = Math.abs(curImgY - startImgY) / ih;

      setBoxROI({ x: Math.max(0, minX), y: Math.max(0, minY), w, h });
    }
  };

  const handleMouseUp = () => {
    setIsDragging(false);
    setIsDrawingBox(false);
  };

  // Zoom on wheel (centered at mouse)
  const handleWheel = (e) => {
    e.preventDefault();
    const zoomFactor = e.deltaY < 0 ? 1.15 : 0.87;
    setScale((prev) => Math.max(0.4, Math.min(5.0, prev * zoomFactor)));
  };

  const resetView = () => {
    setScale(1.0);
    setPan({ x: 0, y: 0 });
    setBoxROI(null);
  };

  // W/L Preset handlers
  const applyPreset = (center, width) => {
    setWindowCenter(center);
    setWindowWidth(width);
  };

  return (
    <div className="canvas-viewport-wrap" ref={containerRef} onWheel={handleWheel}>
      {/* Crosshair & HUD Overlay */}
      <div className="crosshair-hud">
        <span>X: {crosshairPos.x}px</span>
        <span>Y: {crosshairPos.y}px</span>
        <span>Z: {sliceIndex + 1}/{maxSlices}</span>
        <span>Zoom: {Math.round(scale * 100)}%</span>
      </div>

      {/* Main Render Canvas */}
      <canvas
        ref={canvasRef}
        className="pacs-render-canvas"
        style={{ cursor: activeTool === "pan" ? (isDragging ? "grabbing" : "grab") : "crosshair" }}
        onMouseDown={handleMouseDown}
        onMouseMove={handleMouseMove}
        onMouseUp={handleMouseUp}
        onMouseLeave={handleMouseUp}
      />

      {/* Floating Bottom Radiological Dock */}
      <div className="pro-glass-panel floating-bottom-dock">
        {/* Row 1: Slice Scrubber & Cine Loop */}
        <div className="dock-row">
          <div className="pacs-slider-group">
            <span className="slider-label">Slice (Z):</span>
            <input
              type="range"
              min="0"
              max={maxSlices - 1}
              value={sliceIndex}
              onChange={(e) => setSliceIndex(Number(e.target.value))}
              className="pacs-range-input"
            />
            <span className="slider-val-readout">{sliceIndex + 1} / {maxSlices}</span>
          </div>

          <div style={{ display: "flex", gap: "4px", alignItems: "center" }}>
            <button
              className="btn-icon"
              title="Previous Slice (Left Arrow)"
              onClick={() => setSliceIndex((p) => Math.max(0, p - 1))}
            >
              <ChevronLeft size={16} />
            </button>
            <button
              className={`btn-icon ${isPlaying ? "active" : ""}`}
              title={isPlaying ? "Pause Cine Loop" : "Play Cine Loop"}
              onClick={() => setIsPlaying(!isPlaying)}
            >
              {isPlaying ? <Pause size={16} /> : <Play size={16} />}
            </button>
            <button
              className="btn-icon"
              title="Next Slice (Right Arrow)"
              onClick={() => setSliceIndex((p) => Math.min(maxSlices - 1, p + 1))}
            >
              <ChevronRight size={16} />
            </button>
            <button
              className="btn-cad"
              style={{ padding: "4px 8px", fontSize: "10px" }}
              title="Jump to Lesion Center Slice"
              onClick={() => {
                if (meta?.best_slices?.axial !== undefined) {
                  setSliceIndex(meta.best_slices.axial);
                }
              }}
            >
              Key Slice
            </button>
            <button className="btn-icon" title="Reset Viewport" onClick={resetView}>
              <RotateCcw size={15} />
            </button>
            {onRunScan && (
              <button
                className={`btn-run-pacs ${isLoading ? "loading" : ""}`}
                style={{ padding: "4px 10px", fontSize: "10px" }}
                onClick={onRunScan}
                disabled={isLoading}
                title="Run AI Volumetric Segmentation on Current Scan"
              >
                <Activity size={12} />
                <span>{isLoading ? "Running..." : "Run Scan"}</span>
              </button>
            )}
          </div>
        </div>

        {/* Row 2: Contrast Presets, Mask Alpha, and Threshold */}
        <div className="dock-row">
          <div style={{ display: "flex", gap: "6px", alignItems: "center" }}>
            <span className="slider-label" style={{ minWidth: "auto" }}>W/L:</span>
            <button className="btn-cad" onClick={() => applyPreset(50, 400)}>Soft Tissue</button>
            <button className="btn-cad" onClick={() => applyPreset(100, 200)}>High Contrast</button>
            <button className="btn-cad" onClick={() => applyPreset(300, 1000)}>Bone / Dense</button>
            <button className="btn-cad" onClick={() => { setWindowCenter(null); setWindowWidth(null); }}>Default</button>
          </div>

          <div style={{ display: "flex", gap: "16px", alignItems: "center", flex: 1, justifyContent: "flex-end" }}>
            <div className="pacs-slider-group" style={{ maxWidth: "200px" }}>
              <span className="slider-label">Mask:</span>
              <input
                type="range"
                min="0.1"
                max="0.9"
                step="0.05"
                value={maskAlpha}
                onChange={(e) => setMaskAlpha(Number(e.target.value))}
                className="pacs-range-input"
              />
              <span className="slider-val-readout">{Math.round(maskAlpha * 100)}%</span>
            </div>

            <button
              className={`btn-cad ${contourOnly ? "active" : ""}`}
              onClick={() => setContourOnly(!contourOnly)}
              title="Toggle Contour Outline vs Filled Mask"
            >
              Contour
            </button>

            <button
              className={`btn-icon ${showMask ? "active" : ""}`}
              title={showMask ? "Hide Tumor Mask" : "Show Tumor Mask"}
              onClick={() => setShowMask(!showMask)}
            >
              {showMask ? <Eye size={16} /> : <EyeOff size={16} />}
            </button>
          </div>
        </div>
      </div>
    </div>
  );
}
