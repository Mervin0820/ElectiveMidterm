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


