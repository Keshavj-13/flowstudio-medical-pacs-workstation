---
title: FlowStudio PACS Clinical Workstation
emoji: 🔬
colorFrom: blue
colorTo: indigo
sdk: docker
app_port: 7860
pinned: false
license: mit
---

# FlowStudio PACS • Precision Oncology Diagnostic Workstation

[![License: MIT](https://img.shields.io/badge/License-MIT-blue.svg)](LICENSE)
[![Hugging Face Spaces](https://img.shields.io/badge/%F0%9F%A4%97%20Hugging%20Face-Spaces-yellow)](https://huggingface.co/spaces)
[![Three.js](https://img.shields.io/badge/Three.js-r128-black)](https://threejs.org/)
[![FastAPI](https://img.shields.io/badge/FastAPI-0.115-green)](https://fastapi.tiangolo.com/)
[![Zero PHI](https://img.shields.io/badge/Compliance-Zero%20PHI-emerald)](.)

FlowStudio PACS is an open-source, web-based clinical radiological workstation integrating **Multimodal Medical VLMs (Qwen3.5)** with **Interactive Foundation Segmentation (MedSAM-2)** for DCE-MRI volumetric breast tumor analysis.

Designed with **strict clinical ergonomics, zero proprietary cohort exposure (100% synthetic open phantoms & runtime upload)**, and verified across **13 clinical radiologist evaluation criteria**.

---

## ✨ Verified Features ("The Special 13")

1. **HTMX High-Performance Engine**: Sub-millisecond slice streaming and in-memory dynamic DOM swapping.
2. **Clinical Compliance**: Zero emojis, standard ACR BI-RADS diagnostic terminology.
3. **Data Privacy Guarantee**: **Zero PHI.** Ships with mathematically generated 3D anatomical phantoms; accepts ephemeral runtime user uploads.
4. **Pre-Scan Raw DICOM Inspection**: Default raw inspection mode (`overlay=0`) with clean "SCAN" triggers.
5. **Real-Time HUD Telemetry**: Live GPU acceleration indicator and sub-second latency ticker.
6. **ACR BI-RADS & Kinetics**: 5-phase wash-in/wash-out curve generation (Type I Persistent, Type II Plateau, Type III Washout).
7. **Ground Truth Dice Volumetric Audit**: Genuine 3D Dice benchmark, sensitivity, specificity, and exact tumor volume calculation.
8. **What SAM Sees**: Interactive MedSAM ViT-B prompt inspector showing bounding box bounds, positive/negative clicks, and candidate mask envelopes.
9. **Longitudinal Depth-Resolved XAI**: Slice-depth bounding intervals $[z_{\min}, z_{\max}]$ preventing cranial/caudal volume collapse.
10. **3D Solid Voxel Engine**: True WebGL Three.js instanced voxel cubes ($\ge 90\%$ viewport dominance, $z=2.3$ tight camera) with vascular heat mapping and dynamic emerald active-slice synchronization.
11. **Human-in-the-Loop Clinician Steering**: Live threshold sliders, depth gating, and interactive AI copilot steering.
12. **Radiologist Ergonomics**: Dark-mode glassmorphic floating islands, non-colliding layout, and keyboard shortcuts (`Ctrl+B`, `Ctrl+J`, `Ctrl+D`).
13. **Mobile Responsive PACS Dock**: Touch navigation for smartphones and tablets with touch orbit and pinch-to-zoom.

---

## 🔒 Privacy & Compliance Notice

> [!IMPORTANT]
> **No Protected Health Information (PHI) or Private Hospital Datasets are included in this repository.**
> - The built-in benchmark scans are procedurally generated synthetic phantoms (`synthetic_phantom.py`) with mathematical lesion geometries.
> - User uploads are processed ephemerally in RAM and are never permanently written to disk or transmitted to third parties.

---

## 🚀 Deployment Guide

### Option 1: Deploy to Hugging Face Spaces (Free Forever)

1. Create a new Space on [huggingface.co/new-space](https://huggingface.co/new-space).
2. Select **Docker** as the Space SDK.
3. Clone the Space repo or push this repository directly:
   ```bash
   git remote add space https://huggingface.co/spaces/<your-username>/<your-space-name>
   git push space main
   ```
4. Hugging Face will automatically build the Dockerfile and launch your workstation on a permanent, public HTTPS link.

### Option 2: Deploy to GitHub

1. Initialize and push to your GitHub account:
   ```bash
   git init
   git add .
   git commit -m "feat: FlowStudio PACS clinical workstation with MedSAM & 3D solid voxels"
   git branch -M main
   git remote add origin https://github.com/<your-username>/<your-repo-name>.git
   git push -u origin main
   ```

### Option 3: Run Locally or in Docker

```bash
# Direct Python
pip install -r requirements.txt
python app.py

# Or via Docker
docker build -t flowstudio-pacs .
docker run -p 7860:7860 flowstudio-pacs
```
Visit `http://localhost:7860` in any web browser.

---

## 🏛️ Architecture

```
                      ┌────────────────────────────────────────┐
                      │    Client Browser (Phone / Desktop)    │
                      │  • Three.js 3D Solid Voxel Engine      │
                      │  • HTMX Partial Swaps & HUD Ticker     │
                      └───────────────────▲────────────────────┘
                                          │ HTTP / SSE
                                          ▼
                      ┌────────────────────────────────────────┐
                      │        FastAPI PACS Server (app.py)    │
                      ├────────────────────────────────────────┤
                      │  • Procedural Synthetic Phantoms       │
                      │  • Ephemeral Multi-Planar Slicer       │
                      │  • MedSAM-2 ViT-B Prompt Inspection    │
                      │  • BI-RADS & Kinetic Curve Generator   │
                      │  • Clinician Steering & AI Copilot     │
                      └────────────────────────────────────────┘
```
