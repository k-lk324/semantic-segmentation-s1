import sys
from pathlib import Path
import numpy as np
import pytest

REPO_ROOT = Path(__file__).resolve().parent.parent
sys.path.append(str(REPO_ROOT / "scripts"))

from vote_and_reconstruct import (
    softmax,
    map_tile_indices_to_subsampled,
    validate_tile_prediction,
    ensure_output_directories,
)


def test_softmax_properties():
    """Verify softmax outputs sum to 1 across rows and preserve ranking."""
    logits = np.array([[2.0, 1.0, 0.1], [0.0, 3.0, -1.0]], dtype=np.float32)
    probs = softmax(logits)

    assert probs.shape == logits.shape
    np.testing.assert_allclose(probs.sum(axis=1), np.ones(2), rtol=1e-5)
    assert np.all(probs >= 0.0)
    assert np.argmax(probs[0]) == 0
    assert np.argmax(probs[1]) == 1


def test_map_tile_indices_searchsorted_out_of_bounds():
    """Verify that indices larger than any sorted_orig element do not cause IndexError."""
    sorted_orig = np.array([10, 20, 30], dtype=np.int64)
    sorted_order = np.array([0, 1, 2], dtype=np.int64)
    # 5 is smaller, 40 and 100 are strictly larger than 30 (would cause index out of bounds in old code)
    indices = np.array([5, 10, 25, 30, 40, 100], dtype=np.int64)

    valid, mapped_indices = map_tile_indices_to_subsampled(
        sorted_orig=sorted_orig,
        sorted_order=sorted_order,
        indices=indices,
    )

    expected_valid = np.array([False, True, False, True, False, False], dtype=bool)
    np.testing.assert_array_equal(valid, expected_valid)
    np.testing.assert_array_equal(mapped_indices, np.array([0, 2], dtype=np.int64))


def test_map_tile_indices_empty_inputs():
    """Verify empty sorted_orig or empty indices returns gracefully without IndexError."""
    empty_arr = np.array([], dtype=np.int64)
    indices = np.array([1, 2, 3], dtype=np.int64)

    valid, mapped_indices = map_tile_indices_to_subsampled(
        sorted_orig=empty_arr,
        sorted_order=empty_arr,
        indices=indices,
    )
    assert valid.shape == (3,)
    assert not np.any(valid)
    assert mapped_indices.size == 0


def test_validate_tile_prediction_valid():
    """Verify validation passes for matching prediction shape."""
    logits = np.zeros((50, 16), dtype=np.float32)
    validate_tile_prediction(logits, expected_num_points=50, expected_num_classes=16, tile_name="tile_1.pth")


def test_validate_tile_prediction_invalid_dimension():
    """Verify ValueError when logits tensor is not 2D."""
    logits = np.zeros(50, dtype=np.float32)
    with pytest.raises(ValueError, match="invalid dimension"):
        validate_tile_prediction(logits, expected_num_points=50, expected_num_classes=16, tile_name="tile_1.pth")


def test_validate_tile_prediction_classes_mismatch():
    """Verify ValueError when prediction class count differs from expected."""
    logits = np.zeros((50, 4), dtype=np.float32)
    with pytest.raises(ValueError, match="expected 16"):
        validate_tile_prediction(logits, expected_num_points=50, expected_num_classes=16, tile_name="tile_1.pth")


def test_validate_tile_prediction_points_mismatch():
    """Verify ValueError when prediction point count differs from tile point count."""
    logits = np.zeros((45, 16), dtype=np.float32)
    with pytest.raises(ValueError, match="does not match tile point count"):
        validate_tile_prediction(logits, expected_num_points=50, expected_num_classes=16, tile_name="tile_1.pth")


def test_ensure_output_directories(tmp_path):
    """Verify parent directories are created for output files."""
    dest = tmp_path / "nested" / "subfolder" / "output.las"
    assert not dest.parent.exists()

    ensure_output_directories(dest)
    assert dest.parent.exists()
