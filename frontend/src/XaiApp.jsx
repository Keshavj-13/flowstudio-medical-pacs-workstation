/**
 * XAI-Enhanced Medical VLM Studio
 * Incorporates Qwen3.5-0.8B Nuclear Multimodal VLM (85.21% MamaMia / 80.42% Yunnan Golden 3D Dice)
 * Tri-Planar Consensus, Clinical Pathology & BI-RADS Reasoning, and Yunnan Golden Benchmark Explorer.
 */

import { useEffect, useRef, useState } from "react";
import { api } from "./api";
import "./xai.css";
import SliceViewer from "./components/SliceViewer";
import XaiDashboard from "./components/XaiDashboard";
import ModelSelector from "./components/ModelSelector";
import GoldenDatasetPicker from "./components/GoldenDatasetPicker";
import ClinicalReasoningPanel from "./components/ClinicalReasoningPanel";
import {
  UploadCloud, CheckCircle2, AlertCircle, RefreshCw,
  Cpu, Database, Activity, Brain, Award, Sparkles
} from "lucide-react";

export default function XaiApp() {
  const [gpu, setGpu] = useState(null);
  const [gpuErr, setGpuErr] = useState("");
  const [live, setLive] = useState(true);

  // Model Selection
  const [selectedModel, setSelectedModel] = useState("qwen_nuclear_vlm");
  const [models, setModels] = useState([]);

  // File Upload State
  const [files, setFiles] = useState([]);
  const fileRef = useRef(null);

  // Hyperparameters
  const [gpusToUse, setGpusToUse] = useState(1);
  const [threshold, setThreshold] = useState(0.5);
  const [overlap, setOverlap] = useState(0.5);
  const [swBatch, setSwBatch] = useState(4);

  // Golden Cases State
  const [loadingGoldenId, setLoadingGoldenId] = useState(null);

  // Jobs
  const [jobs, setJobs] = useState([]);
  const [submitting, setSubmitting] = useState(false);
  const [uploadProgress, setUploadProgress] = useState(0);
  const [uploadEta, setUploadEta] = useState(null);

  async function refreshGpu() {
    try {
      setGpuErr("");
      setGpu(await api.gpu());
    } catch (e) {
      setGpu(null);
      setGpuErr(String(e?.message || e));
    }
  }

  async function loadModels() {
    try {
      const data = await api.models();
      if (Array.isArray(data)) setModels(data);
    } catch (e) {
      console.warn("Could not load dynamic models list:", e);
    }
  }

  useEffect(() => {
    refreshGpu();
    loadModels();
  }, []);

  useEffect(() => {
    if (!live) return;
    const t = setInterval(refreshGpu, 5000);
    return () => clearInterval(t);
  }, [live]);

  // Poll jobs for status + results/meta
  useEffect(() => {
    const t = setInterval(async () => {
      const active = jobs.filter(j => j.status !== "DONE" && j.status !== "FAILED");
      if (!active.length) return;

      const updates = await Promise.allSettled(active.map(async (j) => {
        const s = await api.status(j.job_id);
        let result = j.result;
        let meta = j.meta;

        if (s.status === "DONE") {
          if (!result) result = await api.result(j.job_id);
          if (!meta) meta = await api.meta(j.job_id);
        }
        return {
          job_id: j.job_id,
          status: s.status,
          error: s.error || null,
          routed_to: s.routed_to,
          result,
          meta
        };
      }));

      setJobs(prev => prev.map(j => {
        const u = updates
          .filter(x => x.status === "fulfilled")
          .map(x => x.value)
          .find(x => x.job_id === j.job_id);
        return u ? { ...j, ...u } : j;
      }));
    }, 1500);

    return () => clearInterval(t);
  }, [jobs]);

  // Handle Golden Benchmark 1-Click Load
  async function handleSelectGoldenCase(caseId) {
    setLoadingGoldenId(caseId);
    try {
      const r = await api.loadGoldenCase(caseId, selectedModel);
      const newJob = {
        job_id: r.job_id,
        filename: r.filename,
        status: r.status || "QUEUED",
        model_type: r.model_type || selectedModel,
        error: null,
        result: null,
        meta: null,
        routed_to: r.routed_to || "GPU Node"
      };
      setJobs(prev => [newJob, ...prev]);
    } catch (err) {
      alert(`Failed to load golden case ${caseId}: ${err.message}`);
    } finally {
      setLoadingGoldenId(null);
    }
  }

  // Handle Custom Scan Submission
  async function submit() {
    if (!files.length) return;
    setSubmitting(true);
    setUploadProgress(0);
    setUploadEta(null);
    try {
      const fd = new FormData();
      for (const f of files) fd.append("files", f, f.name);
      fd.append("gpus", String(gpusToUse));
      fd.append("threshold", String(threshold));
      fd.append("overlap", String(overlap));
      fd.append("sw_batch_size", String(swBatch));
      fd.append("model_type", selectedModel);

      let startTime = Date.now();
      const r = await api.submit(fd, (progressEvent) => {
        if (!progressEvent.lengthComputable) return;
        const percentCompleted = Math.round((progressEvent.loaded * 100) / progressEvent.total);
        setUploadProgress(percentCompleted);
        
        const elapsed = (Date.now() - startTime) / 1000;
        if (elapsed > 1 && percentCompleted > 0 && percentCompleted < 100) {
          const speed = progressEvent.loaded / elapsed;
          const remaining = progressEvent.total - progressEvent.loaded;
          const etaSecs = remaining / speed;
          if (etaSecs && isFinite(etaSecs)) {
            setUploadEta(Math.round(etaSecs));
          }
        }
      });
      
      const newJobs = (r.jobs || []).map(j => ({
        ...j,
        model_type: selectedModel,
        error: null,
        result: null,
        meta: null,
      }));

      setJobs(prev => [...newJobs, ...prev]);
      setFiles([]);
      if (fileRef.current) fileRef.current.value = "";
    } finally {
      setSubmitting(false);
      setUploadProgress(0);
      setUploadEta(null);
    }
  }

  return (
    <div className="page">
      <div className="container">
        
        {/* Header */}
        <div className="header">
          <div>
            <div className="kicker" style={{ color: "var(--accent-cyan)", fontWeight: 700 }}>
              DeepMind Antigravity • Autonomous Oncology VLM
            </div>
            <h1 className="h1">
              Qwen3.5-0.8B Interwoven Multimodal Studio
            </h1>
            <div className="sub">
              3D DCE-MRI Volumetric Reconstruction (&ge;80% Dice) • Tri-Planar Consensus • Clinical Chain-of-Thought &amp; BI-RADS
            </div>
          </div>
          <div className="row" style={{ gap: "1rem" }}>
            <button className="btn ghost" onClick={refreshGpu}>
              <RefreshCw size={16} style={{ marginRight: "6px" }} /> Refresh Hardware
            </button>
            <label className="chk" style={{ cursor: "pointer" }}>
              <input type="checkbox" checked={live} onChange={(e) => setLive(e.target.checked)} />
              Live Polling: {live ? "Active" : "Paused"}
            </label>
          </div>
        </div>

        {/* 1. Model Selector */}
        <ModelSelector
          selectedModel={selectedModel}
          onSelectModel={setSelectedModel}
          models={models}
        />

        {/* 2. Yunnan Golden Benchmark Suite */}
        <GoldenDatasetPicker
          onSelectCase={handleSelectGoldenCase}
          loadingCaseId={loadingGoldenId}
        />

        {/* 3. Main Grid: Upload & Hardware Fleet */}
        <div className="grid">
          
          {/* Custom Upload Analysis Panel */}
          <div className="card">
            <div className="cardHead">
              <h2 className="h2" style={{ display: "flex", alignItems: "center", gap: "8px" }}>
                <Activity size={20} color="var(--accent-cyan)" /> Upload Custom DCE-MRI Volume
              </h2>
            </div>
            
            <div className="uploadArea" onClick={() => fileRef.current?.click()}>
              <UploadCloud size={48} color="var(--text-muted)" style={{ marginBottom: "1rem" }} />
              <div style={{ fontWeight: 600, fontSize: "1.1rem", marginBottom: "0.25rem" }}>
                Drag &amp; Drop Medical Scans
              </div>
              <div className="muted small">Supports 3D/4D NIfTI (.nii, .nii.gz) • Processed by {selectedModel === "qwen_nuclear_vlm" ? "Qwen Nuclear VLM" : "FlexibleUNet"}</div>
              <input
                ref={fileRef}
                type="file"
                multiple
                accept=".nii,.nii.gz"
                onChange={(e) => setFiles(Array.from(e.target.files || []))}
              />
            </div>
            
            {files.length > 0 && (
              <div className="muted small" style={{ marginBottom: "1rem", textAlign: "center" }}>
                Selected: {files.map(f => f.name).join(", ")}
              </div>
            )}

            <div className="row" style={{ justifyContent: "space-between", marginBottom: "1.5rem" }}>
              <label className="mini">
                Contrast Gate
                <input
                  type="number"
                  step="0.05"
                  min="0"
                  max="1"
                  value={threshold}
                  onChange={(e) => setThreshold(Number(e.target.value))}
                />
              </label>
              <label className="mini">
                Window Overlap
                <input
                  type="number"
                  step="0.05"
                  min="0"
                  max="0.95"
                  value={overlap}
                  onChange={(e) => setOverlap(Number(e.target.value))}
                />
              </label>
              <label className="mini">
                SW Batch Size
                <input
                  type="number"
                  min="1"
                  max="32"
                  value={swBatch}
                  onChange={(e) => setSwBatch(Number(e.target.value))}
                />
              </label>
            </div>

            {submitting && uploadProgress > 0 && uploadProgress < 100 && (
              <div style={{ marginBottom: "1rem" }}>
                <div style={{ display: "flex", justifyContent: "space-between", fontSize: "0.8rem", color: "var(--accent-cyan)", marginBottom: "0.25rem" }}>
                  <span>Uploading volume...</span>
                  <span>{uploadProgress}% {uploadEta !== null ? `(~${uploadEta}s left)` : ""}</span>
                </div>
                <div style={{ width: "100%", backgroundColor: "rgba(255,255,255,0.1)", height: "8px", borderRadius: "4px", overflow: "hidden" }}>
                  <div style={{ width: `${uploadProgress}%`, backgroundColor: "var(--accent-cyan)", height: "100%", transition: "width 0.2s" }} />
                </div>
              </div>
            )}

            <button
              className="btn primary"
              style={{ width: "100%" }}
              disabled={!files.length || submitting}
              onClick={submit}
            >
              {submitting ? "Processing Upload..." : `Run Inference with ${selectedModel === "qwen_nuclear_vlm" ? "Qwen Nuclear VLM" : "FlexibleUNet"}`}
            </button>
          </div>

          {/* Infrastructure State */}
          <div className="card">
            <div className="cardHead">
              <h2 className="h2" style={{ display: "flex", alignItems: "center", gap: "8px" }}>
                <Cpu size={20} color="var(--accent-indigo)" /> Hardware Fleet &amp; Pipeline Vitals
              </h2>
            </div>
            {gpuErr ? (
              <div className="warn" style={{ display: "flex", gap: "10px" }}>
                <AlertCircle size={20} /> Connection err: {gpuErr}
              </div>
            ) : (
              <div style={{ display: "flex", flexDirection: "column", height: "calc(100% - 40px)", justifyContent: "space-between" }}>
                <pre className="code" style={{ flex: 1, marginBottom: "0.75rem" }}>
                  {JSON.stringify(gpu, null, 2)}
                </pre>
                <div style={{ backgroundColor: "rgba(255,255,255,0.03)", padding: "8px 12px", borderRadius: "6px", fontSize: "0.75rem", color: "#94a3b8" }}>
                  Active Engine: <strong>{selectedModel === "qwen_nuclear_vlm" ? "Qwen3.5-0.8B LoRA + Tri-Planar Consensus" : "FlexibleUNet 3D ResNet50"}</strong>
                </div>
              </div>
            )}
          </div>

        </div>

        {/* 4. Analytics & Jobs Feed */}
        <div className="card" style={{ marginTop: "1.5rem" }}>
          <div className="cardHead">
            <h2 className="h2" style={{ display: "flex", alignItems: "center", gap: "8px" }}>
              <Database size={20} /> Clinical Cases &amp; Reconstruction Feeds
            </h2>
            <div className="pill" style={{ borderColor: "rgba(255,255,255,0.1)" }}>
              {jobs.length} scans in session
            </div>
          </div>

          {!jobs.length ? (
            <div className="muted" style={{ textAlign: "center", padding: "3rem 0", opacity: 0.5 }}>
              No active analysis jobs. Click a case in the <strong>Yunnan Golden Benchmark</strong> above or upload a scan to begin.
            </div>
          ) : (
            <div className="jobs">
              {jobs.map(j => (
                <div className="job" key={j.job_id} style={{ border: "1px solid rgba(255,255,255,0.12)", marginBottom: "1.5rem", padding: "1.25rem", borderRadius: "10px" }}>
                  <div className="jobTop">
                    <div>
                      <div className="jobTitle" style={{ fontSize: "1.1rem", fontWeight: 700 }}>
                        {j.filename}
                      </div>
                      <div className="muted small" style={{ fontFamily: "monospace" }}>
                        Job ID: {j.job_id.substring(0, 8)}... • Engine: {j.model_type || "qwen_nuclear_vlm"}
                      </div>
                    </div>
                    <div className={`pill ${j.status}`}>
                      {j.status === "DONE" && (
                        <CheckCircle2 size={12} style={{ marginRight: "4px", display: "inline", verticalAlign: "text-bottom" }} />
                      )}
                      {j.status}
                    </div>
                  </div>

                  {j.status === "FAILED" && (
                    <div className="warn" style={{ display: "flex", gap: "8px", marginTop: "0.75rem" }}>
                      <AlertCircle size={16} /> {j.error || "Inference error occurred."}
                    </div>
                  )}

                  {j.status === "DONE" && j.result ? (
                    <>
                      {/* Action Links */}
                      <div className="row" style={{ marginTop: "1rem", gap: "0.75rem" }}>
                        <a className="btn ghost small" href={j.result.mask_url} download>
                          Download 3D Mask (.nii.gz)
                        </a>
                        <a className="btn ghost small" href={j.result.overlay_url} target="_blank" rel="noreferrer">
                          Open Axial Overlay PNG
                        </a>
                        <span style={{ fontSize: "0.85rem", color: "#38bdf8", alignSelf: "center", marginLeft: "auto" }}>
                          Tumor Vol: <strong>{j.result.tumor_volume_ml || (j.result.tumor_volume_voxels * 0.0005).toFixed(2)} mL</strong> ({j.result.tumor_volume_voxels} voxels)
                        </span>
                      </div>

                      {/* ══════ CLINICAL REASONING TRACE ("WHAT IS HAPPENING") ══════ */}
                      <ClinicalReasoningPanel
                        jobId={j.job_id}
                        result={j.result}
                      />

                      {/* ══════ 3D NIIVUE SLICE VIEWER ══════ */}
                      {j.meta ? (
                        <SliceViewer
                          jobId={j.job_id}
                          shapeXYZ={j.meta.shape_xyz}
                          bestSlices={j.meta.best_slices}
                        />
                      ) : (
                        <div className="muted small" style={{ marginTop: "1rem" }}>
                          Loading volumetric viewer bounds…
                        </div>
                      )}

                      {/* ══════ XAI ATTRIBUTION DASHBOARD ══════ */}
                      {j.meta && (
                        <XaiDashboard
                          jobId={j.job_id}
                          shapeXYZ={j.meta.shape_xyz}
                          bestSlices={j.meta.best_slices}
                        />
                      )}
                    </>
                  ) : (
                    j.status !== "FAILED" && (
                      <div className="muted small" style={{ marginTop: "1rem", display: "flex", alignItems: "center", gap: "8px" }}>
                        <Activity size={16} color="var(--accent-cyan)" />
                        VLM inference in progress: extracting spatial tokens, regressing depth bounds, and computing tri-planar consensus...
                      </div>
                    )
                  )}
                </div>
              ))}
            </div>
          )}
        </div>

      </div>
    </div>
  );
}
