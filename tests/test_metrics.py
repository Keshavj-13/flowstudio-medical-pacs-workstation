import numpy as np
import pytest


def manager():
    pytest.importorskip("torch")
    pytest.importorskip("monai")
    from genuine_engine import GenuineInferenceManager
    return GenuineInferenceManager()


def test_physical_volume_and_missing_ground_truth():
    engine = manager()
    prediction = np.ones((2, 2, 2), dtype=np.uint8)
    metrics = engine._calculate_metrics(prediction, None, pixdim=(2, 3, 4))
    assert metrics["pred_volume_ml"] == 0.19
    assert metrics["dice_3d"] is None
    assert metrics["has_gt"] is False


def test_empty_reference_is_still_a_valid_reference():
    engine = manager()
    empty = np.zeros((2, 2, 2), dtype=np.uint8)
    foreground = np.ones_like(empty)
    assert engine._calculate_metrics(foreground, empty)["dice_3d"] == 0.0
    both_empty = engine._calculate_metrics(empty, empty)
    assert both_empty["has_gt"] is True
    assert both_empty["dice_3d"] == 1.0


def test_different_voxel_grids_are_rejected():
    engine = manager()
    with pytest.raises(ValueError, match="same voxel grid"):
        engine._calculate_metrics(np.ones((2, 2, 2)), np.ones((2, 2, 3)))
