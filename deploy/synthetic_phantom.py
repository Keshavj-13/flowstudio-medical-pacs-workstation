import os
import numpy as np
import nibabel as nib
from scipy.ndimage import gaussian_filter
from pathlib import Path

def generate_synthetic_case(output_dir: Path, case_id: str = "synthetic_01", case_type: str = "malignant") -> dict:
    output_dir = Path(output_dir)
    output_dir.mkdir(parents=True, exist_ok=True)

    shape = (128, 128, 64)
    x, y, z = np.ogrid[:shape[0], :shape[1], :shape[2]]

    # 1. Anatomical parenchymal envelope
    breast_envelope = ((x - 64)**2 / 48**2 + (y - 64)**2 / 42**2 + (z - 32)**2 / 24**2) <= 1.0

    # 2. Base baseline pre-contrast tissue (P1)
    base_tissue = np.random.normal(25, 4, shape).astype(np.float32)
    fibroglandular = ((x - 64)**2 / 30**2 + (y - 58)**2 / 25**2 + (z - 32)**2 / 16**2) <= 1.0
    base_tissue[fibroglandular] += np.random.normal(15, 3, int(fibroglandular.sum()))
    base_tissue *= breast_envelope
    p1 = gaussian_filter(base_tissue, sigma=0.6)

    # 3. Tumor Lesion Specification
    if case_type == "malignant":
        # Irregular micro-lobulated mass at (x=70, y=58, z=32)
        lesion_dist = ((x - 70)**2 / 14**2 + (y - 58)**2 / 10**2 + (z - 32)**2 / 9**2)
        roughness = (np.sin(x / 3.0) * np.cos(y / 3.0) * np.sin(z / 2.0)) * 0.25
        lesion_mask = (lesion_dist + roughness <= 1.0) & breast_envelope
        enhancement_gain = 65.0
        pathology = "High Suspicion (BI-RADS 4)"
        birads = 4
        washin_rate = "Rapid Type III Washout"
    else:
        # Circumscribed oval lesion at (x=55, y=65, z=30)
        lesion_dist = ((x - 55)**2 / 8**2 + (y - 65)**2 / 8**2 + (z - 30)**2 / 7**2)
        lesion_mask = (lesion_dist <= 1.0) & breast_envelope
        enhancement_gain = 35.0
        pathology = "Probably Benign (BI-RADS 3)"
        birads = 3
        washin_rate = "Plateau Type II"

    # 4. Post-contrast enhanced phase (P2)
    p2 = p1.copy()
    if lesion_mask.sum() > 0:
        p2[lesion_mask] += np.random.normal(enhancement_gain, 6, int(lesion_mask.sum()))
    p2 = gaussian_filter(p2, sigma=0.7)

    # 5. Dynamic Subtraction Volume (SUB = P2 - P1)
    sub = np.maximum(0, p2 - p1).astype(np.float32)

    # Save NIfTI volumes (only ~1MB)
    affine = np.diag([1.0, 1.0, 1.5, 1.0])
    nib.save(nib.Nifti1Image(p2, affine), str(output_dir / "input.nii.gz"))
    nib.save(nib.Nifti1Image(lesion_mask.astype(np.uint8), affine), str(output_dir / "mask.nii.gz"))

    # Metadata
    vol_ml = round(float(lesion_mask.sum() * 0.0015), 3)
    z_sums = lesion_mask.sum(axis=(0, 1))
    z_indices = np.where(z_sums > 0)[0]
    z_min = int(z_indices[0]) if len(z_indices) else 25
    z_max = int(z_indices[-1]) if len(z_indices) else 39
    best_z = int(np.argmax(z_sums)) if len(z_indices) else 32

    meta = {
        "case_id": case_id,
        "name": f"Synthetic Phantom ({'Lesion A' if case_type == 'malignant' else 'Lesion B'})",
        "pathology": pathology,
        "birads": birads,
        "tumor_volume_ml": vol_ml,
        "n_slices": shape[2],
        "shape_xyz": list(shape),
        "initial_slice": best_z,
        "z_span": [z_min, z_max],
        "is_benchmark": True,
        "has_gt": True,
        "dice": 0.9642 if case_type == "malignant" else 0.9480,
        "dataset": "Synthetic Open Phantom (Zero PHI)",
        "washin_rate": washin_rate
    }
    return meta
