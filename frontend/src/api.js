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

export const api = {
  gpu: () => req("/api/gpu"),
  health: () => req("/api/health"),
  models: () => req("/api/models"),
  goldenCases: () => req("/api/golden_cases"),
  loadGoldenCase: (caseId, modelType = "qwen_nuclear_vlm") =>
    req(`/api/golden_cases/${caseId}/load?model_type=${modelType}`, { method: "POST" }),
  submit: (fd, onProgress) => {
    return new Promise((resolve, reject) => {
      const xhr = new XMLHttpRequest();
      xhr.open("POST", "/api/jobs/submit", true);
      if (onProgress && xhr.upload) xhr.upload.onprogress = onProgress;
      xhr.onload = () => {
        let data;
        try { data = JSON.parse(xhr.responseText); } catch { data = xhr.responseText; }
        if (xhr.status >= 400) reject(new Error(data.error || xhr.statusText));
        else resolve(data);
      };
      xhr.onerror = () => reject(new Error("Network Error"));
      xhr.send(fd);
    });
  },
  status: (id) => req(`/api/jobs/${id}/status`),
  result: (id) => req(`/api/jobs/${id}/result`),
  reasoning: (id) => req(`/api/jobs/${id}/reasoning`),
  meta: (id) => req(`/api/jobs/${id}/meta`),
  triplanar: (id) => req(`/api/jobs/${id}/triplanar`),
  volumeStack: (id) => req(`/api/jobs/${id}/volume_stack`),
  kineticCurve: (id, params = {}) =>
    req(`/api/jobs/${id}/kinetic_curve?${new URLSearchParams(params).toString()}`),
  prompt: (id, promptText, modelType = "qwen_nuclear_vlm") =>
    req(`/api/jobs/${id}/prompt`, {
      method: "POST",
      headers: { "Content-Type": "application/json" },
      body: JSON.stringify({ prompt: promptText, model_type: modelType })
    }),
  promptHistory: (id) => req(`/api/jobs/${id}/prompt_history`),
  imageUrl: (id) => `/api/jobs/${id}/image`,
  maskUrl: (id) => `/api/jobs/${id}/mask`,
  sliceUrl: (id, params = {}) => {
    const p = new URLSearchParams();
    for (const [k, v] of Object.entries(params)) {
      if (v !== undefined && v !== null) p.append(k, String(v));
    }
    return `/api/jobs/${id}/slice?${p.toString()}`;
  },
};
