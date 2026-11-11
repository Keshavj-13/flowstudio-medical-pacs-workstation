# FlowStudio PACS

A FastAPI/HTMX workstation for viewing NIfTI volumes across axial, coronal, and sagittal planes and running locally configured segmentation models.

## Start the volume viewer

Requires Python 3.10 or newer.

```bash
python3 -m venv .venv
.venv/bin/python -m pip install -r requirements.txt
.venv/bin/python -m uvicorn server:app --host 0.0.0.0 --port 8070
```

Open port 8070. The case library reports which configured volumes are actually mounted. A fresh checkout can start without the research repositories or GPU weights. Upload a .nii or .nii.gz file to use viewing mode; uploaded volumes do not infer an organ or modality automatically.

The built-in catalog points to the existing server's dataset locations. Cloning the code does not copy these datasets or model checkpoints.

## Enable segmentation

Use the server's established CUDA/PyTorch environment. Install requirements-inference.txt only after choosing the PyTorch build appropriate to that server. These paths are configurable:

| Environment variable | Default | Required content |
| --- | --- | --- |
| BEATNET_DIR | /workspace/beatnet_v3_asymptotic_ablation | cice_beatnet.py and checkpoints/best_cice_multitask.pth, or checkpoints/best_real_beatnet.pth |
| SEG_DEMO_DIR | /workspace/seg_demo | best_model2020.pth for the legacy model |
| VLM_DIR | /workspace/qwen_interwoven_nuclear_80dice_training | Reserved research path; no working VLM service connection in this workstation |
| FLOWSTUDIO_JOBS_DIR | jobs/ beside server.py | Writable runtime storage for uploads, masks, and metrics |

The model selector exposes CICE-BEATNet and Legacy BEATNet. Missing dependencies or weights produce an actionable error. The root engine's historical VLM implementation uses image heuristics rather than calling a language model, so VLM and VLM consensus are rejected by the workstation API.

The dedicated medical_reasoning_studio is a separate application. Its code and models are not included here. See [deployment status and handoff](docs/DEPLOYMENT.md).

## Results

The result panel uses returned voxel counts, volume, inference duration, and reference overlap. Dice and IoU display N/A when a reference mask is unavailable. Prediction overlays and meshes never substitute a reference mask. Saved prediction NIfTI files retain the input affine. The older diagnostic_dossier.html remains as a historical template and is not used by the scan route.

## Endpoints

| Method | Endpoint | Purpose |
| --- | --- | --- |
| GET | /healthz | Viewer liveness, independent of model readiness |
| GET | /cases | Configured case catalog with availability |
| GET | /cohort-drawer | Searchable study library |
| POST | /cases/{case_id}/select | Mount a configured volume |
| POST | /cases/upload | Upload NIfTI, up to 512 MiB |
| GET | /cases/{case_id}/slice | PNG slice; plane, index, window/level, LUT and overlays |
| POST | /cases/{case_id}/scan | Start the selected model workflow |
| GET | /cases/{case_id}/scan-status | Execute inference and render results; format=json returns metrics |
| GET | /api/jobs/{case_id}/mesh3d | Predicted foreground voxel geometry |
| GET | /api/jobs/{case_id}/dice-audit | Reference overlap audit when both masks exist |

Health success does not verify GPU inference. The scan-status endpoint performs inference and should not be used as a health poll.

## Verification

```bash
.venv/bin/python -m pip install pytest httpx playwright
.venv/bin/python -m playwright install chromium
RUN_BROWSER_TESTS=1 .venv/bin/python -m pytest -q tests
```

Metric tests require PyTorch and MONAI. GitHub Actions runs the viewer/API/browser tests and a separate CPU metric job. The browser test saves desktop and mobile screenshots as workflow artifacts. These checks use artificial test volumes; they do not validate the real server's checkpoints or GPU execution.

## Model storage

Keep application source in GitHub and version model releases in private Hugging Face model repositories, or use DVC with your own object storage. See [the recommendation and tradeoffs](docs/MODEL_STORAGE.md).

## Separate synthetic demo

deploy/app.py is a separate synthetic-phantom demonstration. It is not the GPU workstation or the standalone VLM studio.
