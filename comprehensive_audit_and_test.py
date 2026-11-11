#!/usr/bin/env python3
"""
comprehensive_audit_and_test.py
-------------------------------
Systematically tests all four models:
1. cice_beatnet
2. legacy_beatnet
3. qwen_nuclear_vlm
4. dual_consensus

Both:
A. Directly via GenuineInferenceManager
B. Through HTTP API endpoint (/cases/{case_id}/scan-status?model_type={model}&format=json)

Across diverse organ cases:
- brats_01081 (Brain mpMRI)
- spleen_14 (Spleen 3D CT)
- yunnan_19 (Breast DCE-MRI)
- prostate_28 (Prostate mpMRI)
- colon_001 (Colon 3D CT)
"""

import os
import sys
import time
import json
import urllib.request
from pathlib import Path
import numpy as np
import nibabel as nib

# Workstation root
sys.path.insert(0, "/workspace/cice_pacs_workstation")

from server import get_case_meta, get_loaded_volume, get_loaded_mask
from genuine_engine import GenuineInferenceManager

TEST_CASES = ["brats_01081", "spleen_14", "yunnan_19", "prostate_28", "colon_001"]
MODELS = ["cice_beatnet", "legacy_beatnet", "qwen_nuclear_vlm", "dual_consensus"]
BASE_URL = "http://127.0.0.1:8070"

def test_direct():
    print("=" * 80)
    print("PHASE 1: TESTING MODELS DIRECTLY (GenuineInferenceManager)")
    print("=" * 80)
    
    engine = GenuineInferenceManager.get_instance()
    direct_results = []
    
    for case_id in TEST_CASES:
        meta = get_case_meta(case_id)
        organ = meta.get("organ", "breast")
        modality = meta.get("modality", "3d_mri")
        print(f"\n---> Testing Case: {case_id} [Organ: {organ}, Modality: {modality}]")
        
        vol = get_loaded_volume(case_id, phase="P2")
        gt_mask = get_loaded_mask(case_id, use_pred=False)
        sec_vol = None
        if meta.get("secondary_path") and Path(meta["secondary_path"]).exists():
            try:
                s = nib.load(meta["secondary_path"]).get_fdata().astype(np.float32)
                if s.ndim == 4:
                    s = s[..., 0]
                sec_vol = s
            except Exception as e:
                pass
                
        pixdim = None
        if Path(meta.get("image_path", "")).exists():
            try:
                pixdim = nib.load(meta["image_path"]).header.get_zooms()
            except Exception:
                pass

        for model_id in MODELS:
            t0 = time.perf_counter()
            try:
                if model_id == "qwen_nuclear_vlm":
                    res = engine.infer_vlm(vol, organ=organ, modality=modality, secondary_volume=sec_vol, gt_mask=gt_mask, pixdim=pixdim)
                elif model_id == "legacy_beatnet":
                    res = engine.infer_legacy_beatnet(vol, organ=organ, modality=modality, gt_mask=gt_mask, pixdim=pixdim)
                elif model_id == "dual_consensus":
                    res = engine.infer_consensus(vol, organ=organ, modality=modality, secondary_volume=sec_vol, gt_mask=gt_mask, pixdim=pixdim)
                else: # cice_beatnet
                    res = engine.infer_beatnet(vol, organ=organ, modality=modality, secondary_volume=sec_vol, gt_mask=gt_mask, pixdim=pixdim)
                
                elapsed = (time.perf_counter() - t0) * 1000.0
                dice = res.get("genuine_3d_dice", 0.0)
                roi_dice = res.get("roi_dice", dice)
                vol_ml = res.get("tumor_volume_ml", 0.0)
                latency = res.get("latency_ms", elapsed)
                
                print(f"  [{model_id:18s}] Direct Success | 3D Dice: {dice*100:6.2f}% | ROI Dice: {roi_dice*100:6.2f}% | Vol: {vol_ml:7.2f} mL | Latency: {latency:6.1f}ms")
                direct_results.append({
                    "case_id": case_id,
                    "organ": organ,
                    "model_id": model_id,
                    "dice_3d": dice,
                    "roi_dice": roi_dice,
                    "vol_ml": vol_ml,
                    "latency_ms": latency,
                    "direct_success": True,
                    "birads": res.get("birads_category"),
                    "pathology": res.get("diagnostic_pathology")
                })
            except Exception as e:
                print(f"  [{model_id:18s}] Direct FAILED: {str(e)}")
                direct_results.append({
                    "case_id": case_id,
                    "organ": organ,
                    "model_id": model_id,
                    "error": str(e),
                    "direct_success": False
                })
                
    return direct_results

def test_api():
    print("\n" + "=" * 80)
    print("PHASE 2: TESTING MODELS THROUGH HTTP API")
    print("=" * 80)
    
    api_results = []
    for case_id in TEST_CASES:
        meta = get_case_meta(case_id)
        organ = meta.get("organ", "breast")
        print(f"\n---> API Calling Case: {case_id} [Organ: {organ}]")
        
        for model_id in MODELS:
            url = f"{BASE_URL}/cases/{case_id}/scan-status?model_type={model_id}&format=json"
            t0 = time.time()
            try:
                req = urllib.request.Request(url, headers={"Accept": "application/json"})
                with urllib.request.urlopen(req, timeout=180) as resp:
                    status_code = resp.status
                    body = resp.read().decode("utf-8")
                    data = json.loads(body)
                    total_time = (time.time() - t0) * 1000.0
                    
                    dice = data.get("genuine_3d_dice", 0.0)
                    roi_dice = data.get("roi_dice", dice)
                    vol_ml = data.get("pred_volume_ml", 0.0)
                    lat = data.get("latency_ms", 0.0)
                    tot_lat = data.get("total_latency_ms", total_time)
                    
                    print(f"  [{model_id:18s}] API {status_code} OK | 3D Dice: {dice*100:6.2f}% | ROI Dice: {roi_dice*100:6.2f}% | Vol: {vol_ml:7.2f} mL | Inference Latency: {lat:6.1f}ms | Total: {tot_lat:6.1f}ms")
                    api_results.append({
                        "case_id": case_id,
                        "organ": organ,
                        "model_id": model_id,
                        "status_code": status_code,
                        "dice_3d": dice,
                        "roi_dice": roi_dice,
                        "vol_ml": vol_ml,
                        "latency_ms": lat,
                        "total_latency_ms": tot_lat,
                        "api_success": True,
                        "birads": data.get("birads_category"),
                        "pathology": data.get("diagnostic_pathology"),
                        "think_present": bool(data.get("think_markdown"))
                    })
            except Exception as e:
                print(f"  [{model_id:18s}] API FAILED: {str(e)}")
                api_results.append({
                    "case_id": case_id,
                    "organ": organ,
                    "model_id": model_id,
                    "api_success": False,
                    "error": str(e)
                })
    return api_results

if __name__ == "__main__":
    t_start = time.time()
    direct_res = test_direct()
    api_res = test_api()
    
    summary = {
        "timestamp": time.strftime("%Y-%m-%d %H:%M:%S"),
        "direct_results": direct_res,
        "api_results": api_res
    }
    
    out_file = Path("/workspace/cice_pacs_workstation/audit_test_results.json")
    out_file.write_text(json.dumps(summary, indent=2))
    print("\n" + "=" * 80)
    print(f"Testing completed in {time.time() - t_start:.2f}s. Saved to {out_file}")
    print("=" * 80)
