import os
import sys
import cv2
import pytest

# -------------------------------------------------
# Make src folder importable
# -------------------------------------------------
sys.path.append(
    os.path.abspath(
        os.path.join(os.path.dirname(__file__), "../src")
    )
)

import cartoonify


# -------------------------------------------------
# CONSTANT PATHS
# -------------------------------------------------
INPUT_DIR = "input"
OUTPUT_DIR = "output"
IMAGE_NAME = "cartoonify.jpg"


# -------------------------------------------------
# INTEGRATION TEST USING REAL IMAGE
# -------------------------------------------------

def test_cartoonify_real_image_pipeline():
    image_path = os.path.join(INPUT_DIR, IMAGE_NAME)

    # Check image exists in repo
    assert os.path.exists(image_path), "cartoonify.jpg not found in input folder"

    # Load image
    img = cv2.imread(image_path)
    assert img is not None, "Failed to load cartoonify.jpg"

    # Apply cartoonify
    cartoon = cartoonify.cartoonify(img)

    assert cartoon is not None
    assert cartoon.shape == img.shape

    # Save output to output folder
    saved_path = cartoonify.save_image(
        OUTPUT_DIR,
        IMAGE_NAME,
        cartoon
    )

    # Verify output
    assert os.path.exists(saved_path)
    assert saved_path.startswith(OUTPUT_DIR)
