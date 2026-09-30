import { Eye, Palette, Settings, Sliders } from "lucide-react";

const COLORMAPS = ["jet", "hot", "plasma", "inferno", "viridis", "medical"];

export default function XaiControls({
  showMask, setShowMask,
  showGradcam, setShowGradcam,
  showOcclusion, setShowOcclusion,
  maskAlpha, setMaskAlpha,
  gradcamAlpha, setGradcamAlpha,
  occlusionAlpha, setOcclusionAlpha,
  gradcamCmap, setGradcamCmap,
  occlusionCmap, setOcclusionCmap,
  runGradcam, setRunGradcam,
  runOcclusion, setRunOcclusion,
  occPatchSize, setOccPatchSize,
  occStride, setOccStride,
  occFastMode, setOccFastMode,
  xaiReady,
}) {
  return (
    <>
      {/* Overlay toggles */}
      <div className="xai-card">
        <h3 className="xai-card-title">
          <Eye size={16} /> Overlay Controls
        </h3>
        <div className="xai-control-group">
          <label className="xai-toggle">
            <input type="checkbox" checked={showMask} onChange={e => setShowMask(e.target.checked)} />
            <span className="xai-toggle-label">Segmentation Mask</span>
          </label>
          <div className="xai-slider-row">
            <span className="xai-slider-label">Opacity</span>
            <input type="range" min="0" max="1" step="0.05" value={maskAlpha}
              onChange={e => setMaskAlpha(Number(e.target.value))} />
            <span className="xai-slider-value">{(maskAlpha * 100).toFixed(0)}%</span>
          </div>
        </div>

        <div className="xai-control-group">
          <label className="xai-toggle">
            <input type="checkbox" checked={showGradcam}
              onChange={e => setShowGradcam(e.target.checked)}
              disabled={!xaiReady} />
            <span className="xai-toggle-label">Grad-CAM Heatmap</span>
          </label>
          <div className="xai-slider-row">
            <span className="xai-slider-label">Opacity</span>
            <input type="range" min="0" max="1" step="0.05" value={gradcamAlpha}
              onChange={e => setGradcamAlpha(Number(e.target.value))}
              disabled={!xaiReady} />
            <span className="xai-slider-value">{(gradcamAlpha * 100).toFixed(0)}%</span>
          </div>
        </div>

        <div className="xai-control-group">
          <label className="xai-toggle">
            <input type="checkbox" checked={showOcclusion}
              onChange={e => setShowOcclusion(e.target.checked)}
              disabled={!xaiReady} />
            <span className="xai-toggle-label">Occlusion Map</span>
          </label>
          <div className="xai-slider-row">
            <span className="xai-slider-label">Opacity</span>
            <input type="range" min="0" max="1" step="0.05" value={occlusionAlpha}
              onChange={e => setOcclusionAlpha(Number(e.target.value))}
              disabled={!xaiReady} />
            <span className="xai-slider-value">{(occlusionAlpha * 100).toFixed(0)}%</span>
          </div>
        </div>
      </div>

      {/* Colormap Selection */}
      <div className="xai-card">
        <h3 className="xai-card-title">
          <Palette size={16} /> Colormap
        </h3>
        <div className="xai-cmap-row">
          <label className="xai-select-label">
            Grad-CAM
            <select value={gradcamCmap} onChange={e => setGradcamCmap(e.target.value)}
              className="xai-select" disabled={!xaiReady}>
              {COLORMAPS.map(c => <option key={c} value={c}>{c}</option>)}
            </select>
          </label>
          <label className="xai-select-label">
            Occlusion
            <select value={occlusionCmap} onChange={e => setOcclusionCmap(e.target.value)}
              className="xai-select" disabled={!xaiReady}>
              {COLORMAPS.map(c => <option key={c} value={c}>{c}</option>)}
            </select>
          </label>
        </div>
      </div>

      {/* Analysis Parameters */}
      <div className="xai-card">
        <h3 className="xai-card-title">
          <Settings size={16} /> Analysis Parameters
        </h3>
        <div className="xai-control-group">
          <label className="xai-toggle">
            <input type="checkbox" checked={runGradcam}
              onChange={e => setRunGradcam(e.target.checked)} />
            <span className="xai-toggle-label">Enable Grad-CAM</span>
          </label>
          <label className="xai-toggle">
            <input type="checkbox" checked={runOcclusion}
              onChange={e => setRunOcclusion(e.target.checked)} />
            <span className="xai-toggle-label">Enable Occlusion</span>
          </label>
        </div>

        <div className="xai-param-grid">
          <label className="xai-param">
            <span>Patch Size</span>
            <input type="number" min="4" max="64" step="4" value={occPatchSize}
              onChange={e => setOccPatchSize(Number(e.target.value))} />
          </label>
          <label className="xai-param">
            <span>Stride</span>
            <input type="number" min="2" max="32" step="2" value={occStride}
              onChange={e => setOccStride(Number(e.target.value))} />
          </label>
        </div>

        <label className="xai-toggle" style={{marginTop:'0.5rem'}}>
          <input type="checkbox" checked={occFastMode}
            onChange={e => setOccFastMode(e.target.checked)} />
          <span className="xai-toggle-label">Fast Mode (2× stride)</span>
        </label>
      </div>
    </>
  );
}
