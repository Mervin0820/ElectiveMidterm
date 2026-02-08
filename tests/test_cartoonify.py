import os
import cv2
import numpy as np
import pytest
from caroonify import cartoonify, load_images, save_image  


@pytest.fixture
def sample_image(tmp_path):
    """Create a dummy image for testing."""
    img = np.zeros((100, 100, 3), dtype=np.uint8)
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
    assert np.array_equal(loaded_img, img)

def test_save_image_creates_file(sample_image, output_dir):
    _, img = sample_image
    filename = "saved_test.jpg"
    path = save_image(output_dir, filename, img)
    assert os.path.exists(path)
    saved_img = cv2.imread(path)
    assert np.array_equal(saved_img, img)
