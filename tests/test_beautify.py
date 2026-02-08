import os
import numpy as np
import pytest
import cv2
import sys
from pathlib import Path


sys.path.append(str(Path(__file__).resolve().parents[1] / "src"))

from unittest.mock import patch
from beautify import smooth_skin, load_images, save_image, show_resized


def test_smooth_skin_output_shape():
    skin_bgr = cv2.cvtColor(
        np.uint8([[[128, 150, 100]]]),  
        cv2.COLOR_YCrCb2BGR
    )[0, 0]

    img = np.ones((100, 200, 3), dtype=np.uint8)
    img[:, :] = skin_bgr

    result = smooth_skin(img)

    assert isinstance(result, np.ndarray)
    assert result.shape == img.shape
    assert result.dtype == np.uint8

    assert not np.array_equal(result, img)

def test_smooth_skin_none_input():
    with pytest.raises(ValueError):
        smooth_skin(None)

@patch("cv2.imread")
def test_load_images(mock_imread):
    dummy_img = np.zeros((10, 10, 3), dtype=np.uint8)
    mock_imread.return_value = dummy_img

    filenames = ["img1.jpg", "img2.jpg"]
    input_dir = "dummy_dir"
    results = list(load_images(input_dir, filenames))

    assert len(results) == 2
    for name, img in results:
        assert img.shape == dummy_img.shape
        assert np.array_equal(img, dummy_img)

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
    show_resized("test_win", dummy_img, max_height=700)
    mock_imshow.assert_called_once()
