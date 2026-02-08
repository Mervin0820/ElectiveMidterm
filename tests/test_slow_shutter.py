import os
import pytest
import cv2
import numpy as np
from unittest.mock import patch
from pathlib import Path
import sys
sys.path.append(str(Path(__file__).resolve().parents[1] / "src"))
from slow_shutter import apply_slow_shutter, load_images, save_image, show_resized

@pytest.fixture
def sample_image_path():
    # Path to the image in your repository
    return os.path.join("input", "slow_shutter.png")

def test_apply_slow_shutter_real_image(sample_image_path):
    # Load image
    img = cv2.imread(sample_image_path)
    assert img is not None, "Sample image not found in input/slow_shutter.png"

    # Apply slow shutter effect
    result = apply_slow_shutter(img)

    # Basic checks
    assert isinstance(result, np.ndarray)
    assert result.shape == img.shape
    assert result.dtype == np.uint8
    # Result should differ from original
    assert not np.array_equal(result, img)

def test_apply_slow_shutter_none_input():
    with pytest.raises(ValueError):
        apply_slow_shutter(None)

@patch("cv2.imread")
def test_load_images(mock_imread):
    dummy_img = np.zeros((10, 10, 3), dtype=np.uint8)
    mock_imread.return_value = dummy_img

    filenames = ["img1.png", "img2.png"]
    input_dir = "dummy_dir"
    results = list(load_images(input_dir, filenames))

    assert len(results) == 2
    for name, img in results:
        assert img.shape == dummy_img.shape

@patch("cv2.imwrite")
def test_save_image(mock_imwrite, tmp_path):
    dummy_img = np.zeros((10, 10, 3), dtype=np.uint8)
    filename = "test.png"
    output_dir = tmp_path

    mock_imwrite.return_value = True
    path = save_image(output_dir, filename, dummy_img)

    assert path.startswith(str(output_dir))
    assert path.endswith(filename)
    mock_imwrite.assert_called_once()

@patch("cv2.imshow")
def test_show_resized(mock_imshow):
    dummy_img = np.zeros((1000, 1000, 3), dtype=np.uint8)
    # It should resize because height > 700
    show_resized("test_win", dummy_img, max_height=700)
    mock_imshow.assert_called_once()
