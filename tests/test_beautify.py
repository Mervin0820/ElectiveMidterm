import os
import numpy as np
import pytest
import cv2
import sys
from pathlib import Path
sys.path.append(str(Path(__file__).resolve().parents[1] / "src"))
from beautify import smooth_skin, load_images, save_image, show_resized

@pytest.fixture
def sample_image_path():
    return os.path.join("input", "beautify.png")

def test_smooth_skin_with_real_image(sample_image_path):

    img = cv2.imread(sample_image_path)
    assert img is not None, "Sample image not found in input/beautify.png"

  
    result = smooth_skin(img)

  
    assert isinstance(result, np.ndarray)
    assert result.shape == img.shape
    assert result.dtype == np.uint8
    assert not np.array_equal(result, img)
