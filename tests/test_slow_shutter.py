import os
import numpy as np
import pytest
import cv2
import sys
from pathlib import Path

sys.path.append(str(Path(__file__).resolve().parents[1]))
from unittest.mock import patch, MagicMock
from slow_shutter import (
    apply_slow_shutter,
    load_images,
    save_image,
    show_resized
)

def test_apply_slow_shutter_output_shape():
    img = np.ones((100, 200, 3), dtype=np.uint8) * 255
    result = apply_slow_shutter(img, trail_length=10, step=2, direction=1, blend_original=0.5)
    assert result.shape == img.shape
    assert result.dtype == np.uint8
    assert not np.array_equal(result, img)

def test_apply_slow_shutter_none_input():
    with pytest.raises(ValueError):
        apply_slow_shutter(None)

@patch("cv2.imread")
def test_load_images(mock_imread):
    # Mock image data
    dummy_img = np.zeros((10, 10, 3), dtype=np.uint8)
    mock_imread.return_value = dummy_img

    filenames = ["img1.jpg", "img2.jpg"]
    input_dir = "dummy_dir"
    results = list(load_images(input_dir, filenames))

    assert len(results) == 2
    for name, img in results:
        assert img.shape == dummy_img.shape

@patch("cv2.imwrite")
def test_save_image(mock_imwrite, tmp_path):
    dummy_img = np.zeros((10, 10, 3), dtype=np.uint8)
    filename = "test.jpg"
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
