import { useEffect, useRef, useState } from "react";
import { api } from "./api";
import SliceViewer from "./components/SliceViewer";
import { UploadCloud, CheckCircle2, AlertCircle, RefreshCw, Cpu, Database, Activity } from "lucide-react";

export default function App() {
  const [gpu, setGpu] = useState(null);
  const [gpuErr, setGpuErr] = useState("");
  const [live, setLive] = useState(true);

  const [files, setFiles] = useState([]);
  const fileRef = useRef(null);

  const [gpusToUse, setGpusToUse] = useState(1);
  const [threshold, setThreshold] = useState(0.5);
  const [overlap, setOverlap] = useState(0.5);
  const [swBatch, setSwBatch] = useState(4);

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

  useEffect(() => { refreshGpu(); }, []);
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
        return { job_id: j.job_id, status: s.status, error: s.error || null, routed_to: s.routed_to, result, meta };
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

      let startTime = Date.now();
      const r = await api.submit(fd, (progressEvent) => {
        if (!progressEvent.lengthComputable) return;
        const percentCompleted = Math.round((progressEvent.loaded * 100) / progressEvent.total);
        setUploadProgress(percentCompleted);
        
        const elapsed = (Date.now() - startTime) / 1000; // seconds
        if (elapsed > 1 && percentCompleted > 0 && percentCompleted < 100) {
          const speed = progressEvent.loaded / elapsed; // bytes per sec
          const remaining = progressEvent.total - progressEvent.loaded;
          const etaSecs = remaining / speed;
          if(etaSecs && isFinite(etaSecs)) {
             setUploadEta(Math.round(etaSecs));
          }
        }
      });
      
      const newJobs = (r.jobs || []).map(j => ({
        ...j,
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
            <div className="kicker">Neuro-Oncology Imaging</div>
            <h1 className="h1">Medical Segmentation Platform</h1>
            <div className="sub">Automated 3D tumor segmentation using robust GPU-accelerated PyTorch transformers.</div>
          </div>
          <div className="row" style={{ gap: '1rem' }}>
            <button className="btn ghost" onClick={refreshGpu}><RefreshCw size={16} style={{marginRight: '6px'}}/> Refresh Hardware</button>
            <label className="chk" style={{ cursor: 'pointer' }}>
              <input type="checkbox" checked={live} onChange={(e) => setLive(e.target.checked)} />
              Live Polling: {live ? "Active" : "Paused"}
            </label>
          </div>
        </div>

        <div className="grid">
          
          {/* Inference Panel */}
          <div className="card">
            <div className="cardHead">
              <h2 className="h2" style={{display: 'flex', alignItems: 'center', gap: '8px'}}><Activity size={20} color="var(--accent-cyan)"/> New Analysis</h2>
            </div>
            
            <div 
              className="uploadArea"
              onClick={() => fileRef.current?.click()}
            >
              <UploadCloud size={48} color="var(--text-muted)" style={{marginBottom: '1rem'}} />
              <div style={{fontWeight: 600, fontSize: '1.1rem', marginBottom: '0.25rem'}}>Drag & Drop Medical Scans</div>
              <div className="muted small">Supports NIfTI (.nii, .nii.gz) • High-res volumes allowed</div>
              <input ref={fileRef} type="file" multiple accept=".nii,.nii.gz"
                onChange={(e) => setFiles(Array.from(e.target.files || []))}
              />
            </div>
            
            {files.length > 0 && (
              <div className="muted small" style={{ marginBottom: '1rem', textAlign: 'center' }}>
                Selected: {files.map(f => f.name).join(', ')}
              </div>
            )}

            <div className="row" style={{ justifyContent: 'space-between', marginBottom: '1.5rem' }}>
              <label className="mini">
                Probability Threshold
                <input type="number" step="0.05" min="0" max="1" value={threshold} onChange={(e) => setThreshold(Number(e.target.value))} />
              </label>
              <label className="mini">
                Window Overlap
                <input type="number" step="0.05" min="0" max="0.95" value={overlap} onChange={(e) => setOverlap(Number(e.target.value))} />
              </label>
              <label className="mini">
                SW Batch Size
                <input type="number" min="1" max="32" value={swBatch} onChange={(e) => setSwBatch(Number(e.target.value))} />
              </label>
            </div>

            {submitting && uploadProgress > 0 && uploadProgress < 100 && (
              <div style={{marginBottom: '1rem'}}>
                <div style={{display: 'flex', justifyContent: 'space-between', fontSize: '0.8rem', color: 'var(--accent-cyan)', marginBottom: '0.25rem'}}>
                  <span>Uploading large volume...</span>
                  <span>{uploadProgress}% {uploadEta !== null ? `(~${uploadEta}s left)` : ''}</span>
                </div>
                <div style={{ width: '100%', backgroundColor: 'rgba(255,255,255,0.1)', height: '8px', borderRadius: '4px', overflow: 'hidden' }}>
                  <div style={{ width: `${uploadProgress}%`, backgroundColor: 'var(--accent-cyan)', height: '100%', transition: 'width 0.2s' }} />
                </div>
              </div>
            )}

            <button className="btn primary" style={{width: '100%'}} disabled={!files.length || submitting} onClick={submit}>
              {submitting ? "Processing Upload..." : "Start GPU Inference"}
            </button>
          </div>

          {/* Infrastructure State */}
          <div className="card">
            <div className="cardHead">
              <h2 className="h2" style={{display: 'flex', alignItems: 'center', gap: '8px'}}><Cpu size={20} color="var(--accent-indigo)"/> Hardware Fleet</h2>
            </div>
            {gpuErr ? (
              <div className="warn" style={{display: 'flex', gap: '10px'}}><AlertCircle size={20}/> Connection err: {gpuErr}</div>
            ) : (
              <pre className="code" style={{height: 'calc(100% - 40px)'}}>{JSON.stringify(gpu, null, 2)}</pre>
            )}
          </div>

        </div>

        {/* Analytics Feed */}
        <div className="card" style={{ marginTop: '1.5rem' }}>
          <div className="cardHead">
            <h2 className="h2" style={{display: 'flex', alignItems: 'center', gap: '8px'}}><Database size={20}/> Inference Results</h2>
            <div className="pill" style={{borderColor: 'rgba(255,255,255,0.1)'}}>{jobs.length} scans processed</div>
          </div>

          {!jobs.length ? (
            <div className="muted" style={{textAlign: 'center', padding: '3rem 0', opacity: 0.5}}>
              No analysis jobs currently active.
            </div>
          ) : (
            <div className="jobs">
              {jobs.map(j => (
                <div className="job" key={j.job_id}>
                  <div className="jobTop">
                    <div>
                      <div className="jobTitle">{j.filename}</div>
                      <div className="muted small" style={{fontFamily: 'monospace'}}>ID: {j.job_id.substring(0,8)}... Routed: {j.routed_to || "Edge Node"}</div>
                    </div>
                    <div className={`pill ${j.status}`}>
                       {j.status === "DONE" && <CheckCircle2 size={12} style={{marginRight: '4px', display:'inline', verticalAlign:'text-bottom'}} />}
                       {j.status}
                    </div>
                  </div>

                  {j.status === "FAILED" ? <div className="warn" style={{display: 'flex', gap: '8px'}}><AlertCircle size={16}/> {j.error || "Engine fault"}</div> : null}

                  {j.status === "DONE" && j.result ? (
                    <>
                      <div className="row" style={{ marginTop: '1rem' }}>
                        <a className="btn ghost small" href={j.result.mask_url} download>Download Mask (.nii.gz)</a>
                        <a className="btn ghost small" href={j.result.overlay_url} target="_blank" rel="noreferrer">Open Overlay</a>
                      </div>

                      {j.meta ? (
                        <SliceViewer
                          jobId={j.job_id}
                          shapeXYZ={j.meta.shape_xyz}
                          bestSlices={j.meta.best_slices}
                        />
                      ) : (
                        <div className="muted small" style={{marginTop: '1rem'}}>Loading volumetric bounds…</div>
                      )}

                      <details style={{ marginTop: '1rem', cursor: 'pointer' }}>
                        <summary className="muted small" style={{userSelect:'none'}}>Raw Core Metrics</summary>
                        <pre className="code" style={{marginTop:'0.5rem'}}>{JSON.stringify({
                          volume_voxels: j.result.tumor_volume_voxels,
                          resolution: j.meta?.shape_xyz
                        }, null, 2)}</pre>
                      </details>
                    </>
                  ) : (
                    j.status !== "FAILED" ? <div className="muted small" style={{marginTop: '0.5rem'}}>Model is currently estimating segmentation boundaries...</div> : null
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
