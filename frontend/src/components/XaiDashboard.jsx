import { useState, useEffect, useCallback, useRef } from "react";
import { xaiApi } from "../xaiApi";
import XaiControls from "./XaiControls";
import XaiMetrics from "./XaiMetrics";
import XaiVolumeRenderer from "./XaiVolumeRenderer";
import {
  Brain, Scan, Activity, Download, Crosshair,
  ChevronLeft, ChevronRight, Maximize2, ZoomIn, ZoomOut
} from "lucide-react";

const PLANES = ["axial", "coronal", "sagittal"];
const PLANE_LABELS = { axial: "Axial (Z)", coronal: "Coronal (Y)", sagittal: "Sagittal (X)" };

export default function XaiDashboard({ jobId, shapeXYZ, bestSlices }) {
  // ─── XAI State ───
  const [xaiStatus, setXaiStatus] = useState("NOT_STARTED");
  const [metrics, setMetrics] = useState(null);
  const [error, setError] = useState(null);

  // ─── Overlay Controls ───
  const [showMask, setShowMask] = useState(true);
  const [showGradcam, setShowGradcam] = useState(true);
  const [showOcclusion, setShowOcclusion] = useState(false);
  const [maskAlpha, setMaskAlpha] = useState(0.35);
  const [gradcamAlpha, setGradcamAlpha] = useState(0.5);
  const [occlusionAlpha, setOcclusionAlpha] = useState(0.5);
  const [gradcamCmap, setGradcamCmap] = useState("jet");
  const [occlusionCmap, setOcclusionCmap] = useState("hot");

  // ─── Slice Navigation ───
  const maxSlices = {
    axial: shapeXYZ?.[2] || 128,
    coronal: shapeXYZ?.[1] || 128,
    sagittal: shapeXYZ?.[0] || 128,
  };
  const [sliceIndices, setSliceIndices] = useState({
    axial: bestSlices?.z || Math.floor(maxSlices.axial / 2),
    coronal: bestSlices?.y || Math.floor(maxSlices.coronal / 2),
    sagittal: bestSlices?.x || Math.floor(maxSlices.sagittal / 2),
  });

  // ─── View Controls ───
  const [brightness, setBrightness] = useState(100);
  const [contrast, setContrast] = useState(100);
  const [zoom, setZoom] = useState(1);
  const [show3D, setShow3D] = useState(false);

  // ─── Occlusion Probe ───
  const [probeResult, setProbeResult] = useState(null);
  const [probing, setProbing] = useState(false);

  // ─── XAI Analysis Parameters ───
  const [occPatchSize, setOccPatchSize] = useState(16);
  const [occStride, setOccStride] = useState(8);
  const [occFastMode, setOccFastMode] = useState(false);
  const [runGradcam, setRunGradcam] = useState(true);
  const [runOcclusion, setRunOcclusion] = useState(true);

  // ─── Auto timestamps for cache-busting ───
  const [imgTimestamp, setImgTimestamp] = useState(Date.now());

  // ─── Trigger XAI Analysis ───
  const triggerAnalysis = useCallback(async () => {
    setError(null);
    setXaiStatus("PROCESSING");
    try {
      await xaiApi.analyze(jobId, {
        run_gradcam: runGradcam,
        run_occlusion: runOcclusion,
        occlusion_patch_size: occPatchSize,
        occlusion_stride: occStride,
        occlusion_fast_mode: occFastMode,
      });
    } catch (e) {
      setError(e.message);
      setXaiStatus("FAILED");
    }
  }, [jobId, runGradcam, runOcclusion, occPatchSize, occStride, occFastMode]);

  // ─── Poll XAI Status ───
  useEffect(() => {
    if (xaiStatus !== "PROCESSING" && xaiStatus !== "QUEUED") return;
    const interval = setInterval(async () => {
      try {
        const s = await xaiApi.status(jobId);
        setXaiStatus(s.xai_status || "NOT_STARTED");
        if (s.xai_status === "DONE") {
          setImgTimestamp(Date.now());
          try {
            const m = await xaiApi.metrics(jobId);
            setMetrics(m);
          } catch {}
        }
        if (s.xai_status === "FAILED") {
          setError(s.error || "XAI analysis failed");
        }
      } catch {}
    }, 2000);
    return () => clearInterval(interval);
  }, [jobId, xaiStatus]);

  // ─── Slice URL builder ───
  const getSliceUrl = useCallback((plane) => {
    const params = {
      plane,
      index: sliceIndices[plane],
      show_mask: showMask,
      show_gradcam: showGradcam && xaiStatus === "DONE",
      show_occlusion: showOcclusion && xaiStatus === "DONE",
      mask_alpha: maskAlpha,
      gradcam_alpha: gradcamAlpha,
      occlusion_alpha: occlusionAlpha,
      gradcam_cmap: gradcamCmap,
      occlusion_cmap: occlusionCmap,
      _t: imgTimestamp,
    };
    return xaiApi.sliceUrl(jobId, params);
  }, [jobId, sliceIndices, showMask, showGradcam, showOcclusion,
      maskAlpha, gradcamAlpha, occlusionAlpha, gradcamCmap, occlusionCmap,
      xaiStatus, imgTimestamp]);

  // ─── Probe occlusion patch ───
  const probePatch = useCallback(async (plane, clickEvent) => {
    if (xaiStatus !== "DONE" || probing) return;
    setProbing(true);
    try {
      const d = sliceIndices.axial;
      const h = sliceIndices.coronal;
      const w = sliceIndices.sagittal;
      const result = await xaiApi.probe(jobId, { d, h, w, patch_size: occPatchSize });
      setProbeResult(result);
    } catch (e) {
      setProbeResult({ error: e.message });
    }
    setProbing(false);
  }, [jobId, sliceIndices, occPatchSize, xaiStatus, probing]);

  // ─── Slice navigation helpers ───
  const updateSlice = (plane, delta) => {
    setSliceIndices(prev => ({
      ...prev,
      [plane]: Math.max(0, Math.min(maxSlices[plane] - 1, prev[plane] + delta)),
    }));
  };

  const xaiReady = xaiStatus === "DONE";

  return (
    <div className="xai-dashboard">
      {/* Dashboard Header */}
      <div className="xai-header">
        <div className="xai-header-left">
          <Brain size={24} className="xai-header-icon" />
          <div>
            <h2 className="xai-title">Explainability Analysis</h2>
            <p className="xai-subtitle">3D Grad-CAM &amp; Occlusion Sensitivity</p>
          </div>
        </div>
        <div className="xai-header-right">
          <div className={`xai-status-badge ${xaiStatus}`}>
            <Activity size={12} />
            {xaiStatus === "NOT_STARTED" ? "Ready" : xaiStatus}
          </div>
          {xaiStatus === "NOT_STARTED" || xaiStatus === "FAILED" ? (
            <button className="btn primary xai-analyze-btn" onClick={triggerAnalysis}>
              <Scan size={16} /> Run XAI Analysis
            </button>
          ) : null}
        </div>
      </div>

      {error && (
        <div className="xai-error">
          <span>{error}</span>
        </div>
      )}

      {/* Processing indicator */}
      {(xaiStatus === "PROCESSING" || xaiStatus === "QUEUED") && (
        <div className="xai-processing">
          <div className="xai-processing-spinner" />
          <div>
            <div className="xai-processing-title">Computing XAI Maps...</div>
            <div className="xai-processing-sub">
              {runGradcam && "Grad-CAM"}{runGradcam && runOcclusion && " + "}{runOcclusion && "Occlusion"} analysis in progress
            </div>
          </div>
        </div>
      )}

      {/* Main Layout: Split Screen */}
      <div className="xai-split">

        {/* ─── LEFT: Multi-View MRI Viewer ─── */}
        <div className="xai-viewer-panel">
          {/* View controls bar */}
          <div className="xai-viewer-toolbar">
            <div className="xai-toolbar-group">
              <label className="xai-toolbar-label">
                <ZoomIn size={14} />
                <input type="range" min="0.5" max="3" step="0.1" value={zoom}
                  onChange={e => setZoom(Number(e.target.value))} />
                <span>{Math.round(zoom * 100)}%</span>
              </label>
            </div>
            <div className="xai-toolbar-group">
              <label className="xai-toolbar-label">
                Brightness
                <input type="range" min="50" max="200" step="5" value={brightness}
                  onChange={e => setBrightness(Number(e.target.value))} />
                <span>{brightness}%</span>
              </label>
              <label className="xai-toolbar-label">
                Contrast
                <input type="range" min="50" max="200" step="5" value={contrast}
                  onChange={e => setContrast(Number(e.target.value))} />
                <span>{contrast}%</span>
              </label>
            </div>
            <button className="btn ghost xai-3d-toggle" onClick={() => setShow3D(!show3D)}>
              <Maximize2 size={14} /> {show3D ? "Hide 3D" : "Show 3D"}
            </button>
          </div>

          {/* Tri-plane viewer */}
          <div className="xai-planes">
            {PLANES.map(plane => (
              <div className="xai-plane-card" key={plane}>
                <div className="xai-plane-header">
                  <span className="xai-plane-label">{PLANE_LABELS[plane]}</span>
                  <span className="xai-plane-index">
                    {sliceIndices[plane] + 1} / {maxSlices[plane]}
                  </span>
                </div>
                <div className="xai-plane-img-container"
                  style={{
                    filter: `brightness(${brightness}%) contrast(${contrast}%)`,
                    transform: `scale(${zoom})`,
                  }}
                  onClick={(e) => probePatch(plane, e)}
                  title={xaiReady ? "Click to probe occlusion patch" : ""}
                >
                  <img
                    className="xai-plane-img"
                    src={getSliceUrl(plane)}
                    alt={`${plane} slice`}
                    draggable={false}
                  />
                  {/* Crosshair overlay */}
                  <div className="xai-crosshair-h" />
                  <div className="xai-crosshair-v" />
                </div>
                <div className="xai-plane-nav">
                  <button className="xai-nav-btn" onClick={() => updateSlice(plane, -5)}>
                    <ChevronLeft size={14} /><ChevronLeft size={14} style={{marginLeft:'-8px'}}/>
                  </button>
                  <button className="xai-nav-btn" onClick={() => updateSlice(plane, -1)}>
                    <ChevronLeft size={14} />
                  </button>
                  <input
                    type="range"
                    className="xai-slice-slider"
                    min={0}
                    max={maxSlices[plane] - 1}
                    value={sliceIndices[plane]}
                    onChange={e => setSliceIndices(prev => ({
                      ...prev, [plane]: Number(e.target.value),
                    }))}
                  />
                  <button className="xai-nav-btn" onClick={() => updateSlice(plane, 1)}>
                    <ChevronRight size={14} />
                  </button>
                  <button className="xai-nav-btn" onClick={() => updateSlice(plane, 5)}>
                    <ChevronRight size={14} /><ChevronRight size={14} style={{marginLeft:'-8px'}}/>
                  </button>
                </div>
              </div>
            ))}
          </div>

          {/* 3D Volume Renderer - toggle */}
          {show3D && xaiReady && (
            <XaiVolumeRenderer jobId={jobId} shapeXYZ={shapeXYZ} />
          )}
        </div>

        {/* ─── RIGHT: Controls + Metrics ─── */}
        <div className="xai-control-panel">
          {/* XAI Controls */}
          <XaiControls
            showMask={showMask} setShowMask={setShowMask}
            showGradcam={showGradcam} setShowGradcam={setShowGradcam}
            showOcclusion={showOcclusion} setShowOcclusion={setShowOcclusion}
            maskAlpha={maskAlpha} setMaskAlpha={setMaskAlpha}
            gradcamAlpha={gradcamAlpha} setGradcamAlpha={setGradcamAlpha}
            occlusionAlpha={occlusionAlpha} setOcclusionAlpha={setOcclusionAlpha}
            gradcamCmap={gradcamCmap} setGradcamCmap={setGradcamCmap}
            occlusionCmap={occlusionCmap} setOcclusionCmap={setOcclusionCmap}
            runGradcam={runGradcam} setRunGradcam={setRunGradcam}
            runOcclusion={runOcclusion} setRunOcclusion={setRunOcclusion}
            occPatchSize={occPatchSize} setOccPatchSize={setOccPatchSize}
            occStride={occStride} setOccStride={setOccStride}
            occFastMode={occFastMode} setOccFastMode={setOccFastMode}
            xaiReady={xaiReady}
          />

          {/* Metrics Panel */}
          {xaiReady && metrics && (
            <XaiMetrics metrics={metrics} />
          )}

          {/* Probe Result */}
          {probeResult && (
            <div className="xai-card">
              <h3 className="xai-card-title">
                <Crosshair size={16} /> Patch Probe
              </h3>
              {probeResult.error ? (
                <div className="xai-error-inline">{probeResult.error}</div>
              ) : (
                <div className="xai-probe-grid">
                  <div className="xai-probe-item">
                    <span className="xai-probe-label">Original Probability</span>
                    <span className="xai-probe-value">{(probeResult.original_prob * 100).toFixed(2)}%</span>
                  </div>
                  <div className="xai-probe-item">
                    <span className="xai-probe-label">Masked Probability</span>
                    <span className="xai-probe-value">{(probeResult.masked_prob * 100).toFixed(2)}%</span>
                  </div>
                  <div className="xai-probe-item xai-probe-delta">
                    <span className="xai-probe-label">Delta Score</span>
                    <span className={`xai-probe-value ${probeResult.delta > 0 ? 'positive' : 'negative'}`}>
                      {probeResult.delta > 0 ? '+' : ''}{(probeResult.delta * 100).toFixed(3)}%
                    </span>
                  </div>
                  <div className="xai-probe-item">
                    <span className="xai-probe-label">Patch Origin</span>
                    <span className="xai-probe-value mono">
                      [{probeResult.patch_origin?.join(', ')}]
                    </span>
                  </div>
                </div>
              )}
            </div>
          )}

          {/* Downloads */}
          {xaiReady && (
            <div className="xai-card">
              <h3 className="xai-card-title">
                <Download size={16} /> Export XAI Data
              </h3>
              <div className="xai-download-grid">
                <a className="btn ghost xai-dl-btn"
                  href={xaiApi.downloadUrl(jobId, "gradcam")} download>
                  Grad-CAM NIfTI
                </a>
                <a className="btn ghost xai-dl-btn"
                  href={xaiApi.downloadUrl(jobId, "occlusion")} download>
                  Occlusion NIfTI
                </a>
              </div>
            </div>
          )}
        </div>
      </div>
    </div>
  );
}
