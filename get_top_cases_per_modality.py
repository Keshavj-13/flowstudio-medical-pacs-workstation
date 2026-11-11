#!/usr/bin/env python3
"""
get_top_cases_per_modality.py
-----------------------------
Finds 1 to 3 highest scoring validation examples for each organ & modality
from the official CICE-BEATNet validation evaluation.
"""

import sys
import json
import time
from pathlib import Path
from collections import defaultdict
import torch
import numpy as np
from torch.utils.data import DataLoader

# Patch legacy MONAI MetaKeys for unpickling compatibility
from monai.utils.enums import MetaKeys
class LegacyMetaKey(str):
    def __init__(self, val):
        self.value = val
        self.name = val.upper()
for key in ["original_pixdim", "original_affine", "spatial_shape", "filename_or_obj"]:
    if key not in MetaKeys._value2member_map_:
        MetaKeys._value2member_map_[key] = LegacyMetaKey(key)

SCRIPT_DIR = Path("/workspace/beatnet_v3_asymptotic_ablation")
sys.path.insert(0, str(SCRIPT_DIR))

from cice_beatnet import CICE_BEATNet
from cice_concept_dataset import CICEConceptDataset, cice_collate_fn
from evaluate_validation_by_dataset import get_dataset_name, compute_dice_iou

def extract_top_cases():
    device = torch.device("cuda" if torch.cuda.is_available() else "cpu")
    print(f"Loading CICE-BEATNet on {device}...")
    
    ckpt_path = SCRIPT_DIR / "checkpoints" / "best_cice_multitask.pth"
    model = CICE_BEATNet(
        in_channels=2,
        out_channels=1,
        backbone="resnet50",
        attention_type="scse",
        bilateral_asymmetry=True,
        deep_supervision=False
    ).to(device)
    
    ckpt = torch.load(str(ckpt_path), map_location=device, weights_only=False)
    state_dict = ckpt.get("model_state_dict", ckpt)
    model.load_state_dict(state_dict, strict=False)
    model.eval()
    
    val_dataset = CICEConceptDataset(split="val")
    for s in val_dataset.samples:
        s["ds_name"] = get_dataset_name(s)
        
    val_loader = DataLoader(
        val_dataset,
        batch_size=8,
        shuffle=False,
        num_workers=0,  # run in-process to ensure MetaKeys patch applies
        collate_fn=cice_collate_fn
    )
    
    # Store per (organ, modality)
    # key -> list of {id, dataset, dice, iou}
    results = defaultdict(list)
    
    print(f"Evaluating {len(val_dataset)} validation samples...")
    t0 = time.time()
    
    with torch.no_grad():
        for batch_idx, batch in enumerate(val_loader):
            images = batch["image"].to(device)
            labels = batch["label"].to(device)
            modality_id = batch["modality_id"].to(device)
            organ_id = batch["organ_id"].to(device)
            is_aniso = batch["is_anisotropic"].to(device)
            batch_organs = batch["organ"]
            batch_mods = batch["modality"]
            batch_ids = batch["id"]
            
            with torch.amp.autocast("cuda"):
                out = model(
                    images,
                    modality_id=modality_id,
                    organ_id=organ_id,
                    is_anisotropic=is_aniso
                )
            preds = (torch.sigmoid(out["seg_logits"]) > 0.5).float()
            
            for b in range(images.shape[0]):
                d, iou = compute_dice_iou(preds[b, 0], labels[b, 0])
                sample_idx = batch_idx * 8 + b
                item = val_dataset.samples[sample_idx] if sample_idx < len(val_dataset.samples) else {}
                ds = item.get("ds_name", "unknown")
                
                organ = batch_organs[b]
                mod = batch_mods[b]
                cid = str(batch_ids[b])
                
                results[(organ, mod)].append({
                    "case_id": cid,
                    "dataset": ds,
                    "organ": organ,
                    "modality": mod,
                    "dice": round(float(d), 4),
                    "iou": round(float(iou), 4)
                })
                
    print(f"Completed evaluation in {time.time() - t0:.2f}s")
    
    # Sort and pick top 3 for each organ/modality
    top_per_group = {}
    for (organ, mod), case_list in results.items():
        # Sort by dice descending, then iou descending
        sorted_cases = sorted(case_list, key=lambda x: (x["dice"], x["iou"]), reverse=True)
        # Deduplicate by case_id
        seen = set()
        unique_top = []
        for c in sorted_cases:
            if c["case_id"] not in seen:
                seen.add(c["case_id"])
                unique_top.append(c)
            if len(unique_top) == 3:
                break
        top_per_group[f"{organ}_{mod}"] = {
            "organ": organ,
            "modality": mod,
            "total_val_cases": len(case_list),
            "top_cases": unique_top
        }
        
    out_path = Path("/workspace/cice_pacs_workstation/top_scoring_examples_per_modality.json")
    out_path.write_text(json.dumps(top_per_group, indent=2))
    print(f"Saved results to {out_path}")
    
    for key, data in top_per_group.items():
        print(f"\n=================== {data['organ'].upper()} ({data['modality'].upper()}) [Total: {data['total_val_cases']}] ===================")
        for idx, c in enumerate(data['top_cases'], 1):
            print(f"  #{idx} Case: {c['case_id']:<30} | Dataset: {c['dataset']:<25} | Dice: {c['dice']*100:6.2f}% | IoU: {c['iou']*100:6.2f}%")

if __name__ == "__main__":
    extract_top_cases()
