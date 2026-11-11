from io import BytesIO
import numpy as np
import nibabel as nib
import pytest
from PIL import Image
from fastapi import HTTPException


def test_viewer_starts_without_research_checkout(workstation):
    server, client, *_ = workstation
    assert client.get("/healthz").json()["inference_verified"] is False
    assert client.get("/").status_code == 200
    drawer = client.get("/cohort-drawer")
    assert drawer.status_code == 200
    assert "1 of 2 studies available" in drawer.text
    assert "Not mounted" in drawer.text
    cases = client.get("/cases").json()
    assert [item["available"] for item in cases] == [True, False]
    assert all("image_path" not in item for item in cases)


def test_case_selection_and_missing_case_errors(workstation):
    server, client, case, *_ = workstation
    selected = client.post(f"/cases/{case['case_id']}/select")
    assert selected.status_code == 200
    assert "legacy_beatnet" in selected.text
    assert "Multi-organ 3D" in selected.text
    assert 'value="flexible_unet"' not in selected.text
    assert client.post("/cases/missing_volume/select").status_code == 404
    assert client.post("/cases/unknown/select").status_code == 404
    assert client.get("/cases/yunnan_not_a_case/slice").status_code == 404


@pytest.mark.parametrize("plane,size", [("axial", (10, 8)), ("coronal", (8, 6)), ("sagittal", (10, 6))])
def test_slice_planes_and_clamped_indices(workstation, plane, size):
    _, client, case, *_ = workstation
    response = client.get(f"/cases/{case['case_id']}/slice", params={"plane": plane, "index": 9999})
    assert response.status_code == 200
    assert response.headers["content-type"] == "image/png"
    assert Image.open(BytesIO(response.content)).size == size


def test_reference_is_never_a_prediction(workstation):
    server, client, case, *_ = workstation
    assert server.get_loaded_mask(case["case_id"], use_pred=True) is None
    assert server.get_loaded_mask(case["case_id"], use_pred=False).sum() > 0
    assert client.get(f"/api/jobs/{case['case_id']}/mesh3d").json()["voxel_count"] == 0
    assert client.get(f"/api/jobs/{case['case_id']}/dice-audit").json()["has_gt"] is False
    plain = client.get(f"/cases/{case['case_id']}/slice?index=2").content
    prediction = client.get(f"/cases/{case['case_id']}/slice?index=2&overlay=1").content
    reference = client.get(f"/cases/{case['case_id']}/slice?index=2&overlay=1&show_gt=1").content
    assert plain == prediction
    assert reference != prediction


@pytest.mark.parametrize("model", ["qwen_nuclear_vlm", "dual_consensus", "flexible_unet", "unknown"])
def test_unsupported_model_does_not_fall_back(workstation, model):
    _, client, case, *_ = workstation
    assert client.post(f"/cases/{case['case_id']}/scan", data={"model_type": model}).status_code == 422
    assert client.get(f"/cases/{case['case_id']}/scan-status", params={"model_type": model}).status_code == 422


def test_dependency_failure_is_actionable(workstation, monkeypatch):
    server, client, case, *_ = workstation
    def unavailable(_):
        raise HTTPException(503, "Model dependencies unavailable")
    monkeypatch.setattr(server, "get_inference_engine", unavailable)
    response = client.post(f"/cases/{case['case_id']}/scan", data={"model_type": "cice_beatnet"})
    assert response.status_code == 503
    assert "dependencies" in response.json()["detail"]


def test_nifti_upload_and_bad_file_cleanup(workstation):
    server, client, _, image_path, _ = workstation
    uploaded = client.post("/cases/upload", files={"file": ("scan.nii.gz", image_path.read_bytes(), "application/gzip")})
    assert uploaded.status_code == 200
    assert "scan.nii.gz" in uploaded.text
    case_id = next(iter(server.CUSTOM_CASE_REGISTRY))
    assert client.get(f"/cases/{case_id}/slice").status_code == 200
    assert client.post(f"/cases/{case_id}/scan").status_code == 422
    before = set(server.JOBS_DIR.iterdir())
    for filename in ("bad.nii", "bad.txt"):
        assert client.post("/cases/upload", files={"file": (filename, b"invalid")}).status_code == 422
    assert set(server.JOBS_DIR.iterdir()) == before


def test_inferno_colormap_cpu_path(workstation, monkeypatch):
    server, *_ = workstation
    monkeypatch.setattr(server, "HAS_CUPY", False)
    grayscale = np.array([[0.25, 0.5, 0.75]], dtype=np.float32)
    colored = server.apply_lut(grayscale, "inferno")
    assert colored.shape == (1, 3, 3)
    assert not np.array_equal(colored[..., 0], colored[..., 1])


def test_results_without_reference_and_saved_affine(workstation, monkeypatch):
    server, client, case, _, affine = workstation
    case["mask_path"], case["has_gt"] = None, False
    prediction = np.ones((8, 10, 6), dtype=np.uint8)
    result = {
        "model_name": "Fixture model", "pred_mask": prediction, "genuine_3d_dice": None,
        "roi_dice": None, "validation_expected_dice": None, "tumor_volume_ml": 11.52,
        "predicted_depth_slices": [0, 5], "gt_depth_slices": None, "pred_voxels": 480,
        "gt_voxels": 0, "latency_ms": 1.0, "birads_category": 4,
        "diagnostic_pathology": "Fixture", "concepts": {}, "has_gt": False, "iou_3d": None
    }
    class Engine:
        def infer_beatnet(self, *args, **kwargs):
            return result
    monkeypatch.setattr(server, "get_inference_engine", lambda _: Engine())
    response = client.get(f"/cases/{case['case_id']}/scan-status")
    assert response.status_code == 200
    assert "N/A" in response.text
    assert "96.42%" not in response.text
    saved = nib.load(server.JOBS_DIR / case["case_id"] / "pred_mask.nii.gz")
    np.testing.assert_allclose(saved.affine, affine)
    metrics = client.get(f"/cases/{case['case_id']}/scan-status?format=json").json()
    assert metrics["gt_depth_slices"] is None
    assert metrics["genuine_3d_dice"] is None
