# FlowStudio PACS

[![FastAPI](https://img.shields.io/badge/Backend-FastAPI-009688?style=flat&logo=fastapi)](https://fastapi.tiangolo.com)
[![HTMX](https://img.shields.io/badge/Frontend-HTMX-336699?style=flat&logo=htmx)](https://htmx.org)
[![Three.js](https://img.shields.io/badge/3D_Rendering-Three.js-black?style=flat&logo=threedotjs)](https://threejs.org)
[![DICOM-SR](https://img.shields.io/badge/Standard-DICOM--SR_TID_1500-blue?style=flat)](#)

FlowStudio PACS is a web workstation for 3D volumetric multi-planar reconstruction (MPR), streaming vision-language reasoning, and DICOM-SR TID 1500 report generation.

## 1. Interface

### Volumetric navigation and 3D rendering

| Tri-planar scrubbing and colormaps | Autonomous scan and segmentation | 3D WebGL solid voxel orbit |
| :---: | :---: | :---: |
| <img src="assets/01_flowstudio_triplanar_scrub.gif" width="100%" alt="Tri-planar scrubbing" /> | <img src="assets/02_flowstudio_autonomous_think_stream.gif" width="100%" alt="Autonomous scan" /> | <img src="assets/03_flowstudio_3d_webgl_orbit.gif" width="100%" alt="3D WebGL orbit" /> |
| Volumetric slice scrubbing with window/level adjustment and colormaps. | Neural scanline sweep, deep feature extraction, and contour toggling. | Three.js WebGL orbit with synchronized 2D slice cutting plane. |

### Diagnostic intelligence and telemetry

| Vision-language chain-of-thought stream | Continuous cine loop and caliper tool |
| :---: | :---: |
| <img src="assets/04_flowstudio_vlm_chain_of_thought.gif" width="100%" alt="VLM reasoning stream" /> | <img src="assets/05_flowstudio_cine_loop_volumetric.gif" width="100%" alt="Volumetric cine loop and caliper" /> |
| Multimodal reasoning with kinetic analysis and longitudinal interval checks. | 10 FPS continuous cross-sectional sweep and sub-millimeter lesion caliper tool. |

<table align="center" width="100%">
  <tr>
    <td width="50%" align="center">
      <b>Cohort catalog</b><br/>
      <img src="assets/01_workstation_catalog.png" width="100%" />
    </td>
    <td width="50%" align="center">
      <b>BraTS multi-planar dossier and contours</b><br/>
      <img src="assets/02_brats_cice_beatnet_dossier.png" width="100%" />
    </td>
  </tr>
</table>

## 2. Architecture and streaming flow

```mermaid
sequenceDiagram
    autonumber
    actor Clinician as Radiologist
    participant UI as Frontend (HTMX / Three.js)
    participant Server as FastAPI Core
    participant GPU as GPU Inference Engine

    Clinician->>UI: Selects patient case (e.g. Case 19)
    UI->>Server: GET /cases/yunnan_19
    Server-->>UI: Serves case dossier and slice stack
    Clinician->>UI: Selects model and clicks Start Scan
    UI->>Server: POST /cases/yunnan_19/scan (SSE channel)
    Server->>GPU: Dispatches volume to GPU
    loop Telemetry streaming
        GPU-->>Server: Reasoning token: wash-in asymmetry detected
        Server-->>UI: Server-Sent Event push
    end
    GPU->>GPU: Generates 3D volumetric contours
    GPU-->>Server: Mask tensor and classification
    Server-->>UI: Updates canvas with 3D mesh
    Clinician->>UI: Reviews contours and clicks Export DICOM-SR
    UI->>Server: GET /cases/yunnan_19/export-sr
    Server-->>Clinician: Returns TID 1500 DICOM file
```

## 3. Supported cohorts and models

### Clinical cohorts
* Breast DCE-MRI: Yunnan cohort (`yunnan_1`, `yunnan_2`, `yunnan_9`, `yunnan_19`), DUKE cohort (`DUKE_211`).
* Neuro-oncology: BraTS 2021 brain mpMRI (`#01081`, `#00251`, `BRATS_239`).
* Abdominal and body CT: MSD Task09 Spleen, MSD Task05 Prostate, MSD Task03 Liver, MSD Task07 Pancreas, MSD Task06 Lung, DeepLesion.

### Model backbones
* BEATNet CICE: Multi-organ 3D segmentation model.
* Qwen3.5 Multimodal VLM (0.8B) with MedSAM-2: Multimodal triage with reasoning traces.
* Dual-model consensus: Agreement filtering across segmentations.

## 4. API endpoints

| Endpoint | Method | Params | Description |
| :--- | :---: | :--- | :--- |
| `/cases` | `GET` | None | Lists available cases with modality and metadata. |
| `/cases/{case_id}` | `GET` | `phase: str`, `show_gt: int` | Renders case dossier and active slice stage. |
| `/cases/{case_id}/scan` | `POST` | `model_type: str` | Dispatches scan and streams reasoning output. |
| `/cases/{case_id}/status` | `GET` | None | Returns progress percentage and Dice scores. |
| `/cases/{case_id}/export-sr` | `GET` | None | Downloads standardized DICOM-SR TID 1500 report. |

## 5. Quickstart

```bash
# Clone the repository
git clone https://github.com/Keshavj-13/flowstudio-medical-pacs-workstation.git
cd flowstudio-medical-pacs-workstation

# Install dependencies
pip install -r requirements.txt

# Run the workstation server
uvicorn server:app --host 0.0.0.0 --port 8070

# Or run the self-contained deployment server
python deploy/app.py
```
