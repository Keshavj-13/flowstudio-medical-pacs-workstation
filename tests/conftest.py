from pathlib import Path
import sys
import numpy as np
import nibabel as nib
import pytest
from fastapi.testclient import TestClient

sys.path.insert(0, str(Path(__file__).resolve().parents[1]))
import server


@pytest.fixture
def workstation(tmp_path, monkeypatch):
    for name in ("VOLUME_CACHE", "MASK_CACHE", "PNG_CACHE", "CASE_META_CACHE", "CUSTOM_CASE_REGISTRY"):
        monkeypatch.setattr(server, name, {})
    jobs = tmp_path / "jobs"
    jobs.mkdir()
    monkeypatch.setattr(server, "JOBS_DIR", jobs)
    image_path = tmp_path / "volume.nii.gz"
    mask_path = tmp_path / "reference.nii.gz"
    affine = np.diag([2.0, 3.0, 4.0, 1.0])
    volume = np.arange(8 * 10 * 6, dtype=np.float32).reshape(8, 10, 6)
    mask = np.zeros_like(volume, dtype=np.uint8)
    mask[2:5, 3:7, 1:4] = 1
    nib.save(nib.Nifti1Image(volume, affine), image_path)
    nib.save(nib.Nifti1Image(mask, affine), mask_path)
    case = {
        "case_id": "fixture_brain", "name": "Test brain volume", "organ": "brain",
        "modality": "3d_mri", "pathology": "Test fixture", "image_path": str(image_path),
        "mask_path": str(mask_path), "has_gt": True, "shape_xyz": [8, 10, 6],
        "n_slices": 6, "initial_slice": 3, "dataset": "Test fixture",
        "available_phases": ["T1CE"], "tumor_volume_ml": None
    }
    missing = dict(case, case_id="missing_volume", name="Unavailable fixture",
                   image_path=str(tmp_path / "missing.nii.gz"))
    monkeypatch.setattr(server, "MULTI_ORGAN_COHORTS", [case, missing])
    with TestClient(server.app) as client:
        yield server, client, case, image_path, affine
