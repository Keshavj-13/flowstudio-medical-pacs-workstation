import { useEffect, useRef, useState } from "react";
import { api } from "../api";
import { Layers, Eye } from "lucide-react";
import { Niivue } from "@niivue/niivue";

export default function SliceViewer({ jobId }) {
  const canvasRef = useRef(null);
  const nvRef = useRef(null);
  const [overlay, setOverlay] = useState(true);
  const [alpha, setAlpha] = useState(0.40);
  const [error, setError] = useState(null);

  useEffect(() => {
    if (!canvasRef.current || !jobId) return;

    const initializeNiivue = async () => {
      try {
        // Initialize Niivue with some default options for smooth rendering
        const nv = new Niivue({ 
          dragAndDropEnabled: false,
          backColor: [0, 0, 0, 1], // Black background by default
          show3Dcrosshair: true,
        });
        
        nv.attachToCanvas(canvasRef.current);
        nvRef.current = nv;

        const imageUrl = api.imageUrl(jobId);
        const maskUrl = api.maskUrl(jobId);

        // Add fake extensions to URLs so Niivue can infer the format from the URL
        const imageVolumeUrl = imageUrl + '?ext=.nii.gz';
        const maskVolumeUrl = maskUrl + '?ext=.nii.gz';

        // Load primary volume (grayscale) and mask overlay (red)
        await nv.loadVolumes([
          { url: imageVolumeUrl, name: "image.nii.gz", colormap: "gray" },
          { url: maskVolumeUrl, name: "mask.nii.gz", colormap: "red", opacity: alpha }
        ]);
      } catch (err) {
        console.error("Niivue failed to load volumes:", err);
        setError("Failed to load 3D viewer. Please ensure your files are valid NIfTI formats.");
      }
    };

    initializeNiivue();

    // Cleanup on unmount
    return () => {
      if (nvRef.current) {
        nvRef.current.volumes.length = 0; // clear memory
      }
    };
  }, [jobId]); // Only reinit if jobId changes

  // Update mask opacity and visibility dynamically
  useEffect(() => {
    if (nvRef.current && nvRef.current.volumes.length > 1) {
      nvRef.current.setOpacity(1, overlay ? alpha : 0.0);
    }
  }, [alpha, overlay]);

  return (
    <div className="viewer" style={{ marginTop: '2rem' }}>
      <div className="viewerTop">
        <div className="viewerTitle"><Layers size={22} color="var(--accent-cyan)" /> 3D Niivue Explorer</div>
        <div className="viewerControls">
          <label className="chk">
            <input type="checkbox" checked={overlay} onChange={(e) => setOverlay(e.target.checked)} />
            <Eye size={16} /> Show Overlay
          </label>
          <label className="mini" style={{ width: '120px' }}>
            <div style={{ display: 'flex', justifyContent: 'space-between' }}>
              <span>Opacity</span><span style={{ color: 'var(--accent-cyan)' }}>{alpha.toFixed(2)}</span>
            </div>
            <input type="range" min="0" max="1" step="0.05" value={alpha} onChange={(e) => setAlpha(Number(e.target.value))} style={{ margin: 0 }} />
          </label>
        </div>
      </div>

      <div style={{ width: '100%', height: '500px', backgroundColor: '#000', borderRadius: '8px', overflow: 'hidden', position: 'relative' }}>
        {error ? (
          <div style={{ color: 'red', padding: '20px', textAlign: 'center' }}>{error}</div>
        ) : (
          <canvas ref={canvasRef} style={{ width: '100%', height: '100%', display: 'block' }}></canvas>
        )}
      </div>
    </div>
  );
}
