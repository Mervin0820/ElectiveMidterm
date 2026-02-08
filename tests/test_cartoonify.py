import os
import numpy as np
import pytest
import cv2
import sys
from pathlib import Path
sys.path.append(str(Path(__file__).resolve().parents[1] / "src"))
from cartoonify import cartoonify, load_images, save_image  


@pytest.fixture
def sample_image(tmp_path):
    """Create a dummy image with content for testing."""
    img = np.zeros((100, 100, 3), dtype=np.uint8)
    cv2.rectangle(img, (25, 25), (75, 75), (255, 255, 255), -1)  # white square
    file_path = tmp_path / "test.jpg"
    cv2.imwrite(str(file_path), img)
    return str(file_path), img

@pytest.fixture
def output_dir(tmp_path):
    return tmp_path / "output"


def test_cartoonify_returns_image(sample_image):
    _, img = sample_image
    result = cartoonify(img)
    assert isinstance(result, np.ndarray)
    assert result.shape == img.shape
    assert np.any(result != 0)

def test_cartoonify_none_input():
    with pytest.raises(ValueError):
        cartoonify(None)

def test_load_images_yields_image(sample_image, tmp_path):
    file_path, img = sample_image
    filenames = [os.path.basename(file_path)]
    input_dir = tmp_path
    results = list(load_images(input_dir, filenames))
    assert len(results) == 1
    name, loaded_img = results[0]
    assert name == filenames[0]
    # Use allclose instead of array_equal
    assert np.allclose(loaded_img, img, atol=1)

def test_save_image_creates_file(sample_image, output_dir):
    _, img = sample_image
    filename = "saved_test.png"  # use PNG to reduce compression artifacts
    path = save_image(output_dir, filename, img)
    assert os.path.exists(path)
    saved_img = cv2.imread(path)
    assert np.allclose(saved_img, img, atol=1)
