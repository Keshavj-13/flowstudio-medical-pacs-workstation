/**
 * XAI API Client
 * 
 * Extends the existing API with XAI-specific endpoints.
 * Keeps the original api.js completely untouched.
 */

async function req(path, options = {}) {
  const res = await fetch(path, { ...options });
  const text = await res.text();
  let data;
  try { data = text ? JSON.parse(text) : null; } catch { data = text; }
  if (!res.ok) {
    const msg = typeof data === "string" ? data : (data?.error || data?.detail || res.statusText);
    throw new Error(`${options.method || "GET"} ${path} failed: ${res.status} ${msg}`);
  }
  return data;
}

export const xaiApi = {
  /** Trigger XAI analysis for a completed job */
  analyze: (id, params = {}) =>
    req(`/api/jobs/${id}/xai/analyze?${new URLSearchParams(params)}`, { method: "POST" }),

  /** Get XAI status */
  status: (id) => req(`/api/jobs/${id}/xai/status`),

  /** Get XAI metrics */
  metrics: (id) => req(`/api/jobs/${id}/xai/metrics`),

  /** Get XAI slice image URL with overlay params */
  sliceUrl: (id, params) =>
    `/api/jobs/${id}/xai/slice?${new URLSearchParams(params).toString()}`,

  /** Download XAI NIfTI heatmap */
  downloadUrl: (id, kind) => `/api/jobs/${id}/xai/download/${kind}`,

  /** Probe a single occlusion patch */
  probe: (id, params) =>
    req(`/api/jobs/${id}/xai/probe?${new URLSearchParams(params)}`),

  /** Get available colormaps */
  colormaps: () => req("/api/xai/colormaps"),

  /** Get raw 3D volume data for rendering */
  volumeData: async (id, kind) => {
    const res = await fetch(`/api/jobs/${id}/xai/volume/${kind}`);
    if (!res.ok) throw new Error(`Failed to fetch ${kind} volume`);
    const shape = (res.headers.get("X-Shape") || "").split(",").map(Number);
    const buffer = await res.arrayBuffer();
    return { shape, data: new Float32Array(buffer) };
  },
};
