import { BarChart3, TrendingUp, Target, Percent } from "lucide-react";

export default function XaiMetrics({ metrics }) {
  if (!metrics) return null;

  const tumorVol = metrics.tumor_volume_voxels || 0;
  const totalVox = metrics.total_voxels || 1;
  const tumorPct = ((tumorVol / totalVox) * 100).toFixed(3);

  return (
    <div className="xai-card">
      <h3 className="xai-card-title">
        <BarChart3 size={16} /> Quantitative Metrics
      </h3>

      {/* Tumor Volume */}
      <div className="xai-metric-section">
        <div className="xai-metric-header">
          <Target size={14} /> Tumor Volume
        </div>
        <div className="xai-metric-row">
          <span className="xai-metric-label">Volume</span>
          <span className="xai-metric-value">
            {tumorVol.toLocaleString()} voxels
          </span>
        </div>
        <div className="xai-metric-row">
          <span className="xai-metric-label">Volume (mm³)</span>
          <span className="xai-metric-value">
            {tumorVol.toLocaleString()} mm³
          </span>
        </div>
        <div className="xai-metric-row">
          <span className="xai-metric-label">Tissue Fraction</span>
          <span className="xai-metric-value">{tumorPct}%</span>
        </div>
      </div>

      {/* Grad-CAM Metrics */}
      {metrics.gradcam_mean_inside_tumor !== undefined && (
        <div className="xai-metric-section">
          <div className="xai-metric-header">
            <TrendingUp size={14} /> Grad-CAM Analysis
          </div>
          <div className="xai-metric-row">
            <span className="xai-metric-label">Mean Intensity (Inside Tumor)</span>
            <span className="xai-metric-value highlight-cyan">
              {metrics.gradcam_mean_inside_tumor.toFixed(4)}
            </span>
          </div>
          <div className="xai-metric-row">
            <span className="xai-metric-label">Mean Intensity (Outside Tumor)</span>
            <span className="xai-metric-value">
              {metrics.gradcam_mean_outside_tumor.toFixed(4)}
            </span>
          </div>

          {/* Visual bar showing inside vs outside */}
          <div className="xai-metric-bar-container">
            <div className="xai-metric-bar-label">
              <Percent size={12} /> Activation Distribution
            </div>
            <div className="xai-metric-bar">
              <div
                className="xai-metric-bar-fill inside"
                style={{ width: `${metrics.gradcam_pct_inside_tumor}%` }}
              />
            </div>
            <div className="xai-metric-bar-legend">
              <span className="xai-legend-inside">
                Inside: {metrics.gradcam_pct_inside_tumor.toFixed(1)}%
              </span>
              <span className="xai-legend-outside">
                Outside: {metrics.gradcam_pct_outside_tumor.toFixed(1)}%
              </span>
            </div>
          </div>
        </div>
      )}

      {/* Occlusion Metrics */}
      {metrics.occlusion_mean_inside_tumor !== undefined && (
        <div className="xai-metric-section">
          <div className="xai-metric-header">
            <BarChart3 size={14} /> Occlusion Sensitivity
          </div>
          <div className="xai-metric-row">
            <span className="xai-metric-label">Mean Importance (Inside)</span>
            <span className="xai-metric-value highlight-orange">
              {metrics.occlusion_mean_inside_tumor.toFixed(4)}
            </span>
          </div>
          <div className="xai-metric-row">
            <span className="xai-metric-label">Mean Importance (Outside)</span>
            <span className="xai-metric-value">
              {metrics.occlusion_mean_outside_tumor.toFixed(4)}
            </span>
          </div>
          <div className="xai-metric-row">
            <span className="xai-metric-label">Sensitivity Score</span>
            <span className={`xai-metric-value ${metrics.occlusion_sensitivity_score > 0 ? 'highlight-green' : 'highlight-red'}`}>
              {metrics.occlusion_sensitivity_score > 0 ? '+' : ''}{metrics.occlusion_sensitivity_score.toFixed(4)}
            </span>
          </div>
        </div>
      )}
    </div>
  );
}
