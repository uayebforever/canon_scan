import pytest

import canon_scan

import os
import numpy as np

TEST_DATA_DIRECTORY=os.path.join(os.environ["HOME"], "Desktop", "Plans")


def test_canon_image_open():
    image_filename = os.path.join(TEST_DATA_DIRECTORY, "IMG_0010.JPG")
    print(image_filename)
    img = canon_scan.CanonImage(image_filename)

    assert isinstance(img.grayscale_data, np.ndarray)


def test_flatfield():
    cs = canon_scan.CalibrationSet()

    cs.flat_field = canon_scan.CanonImage("IMG_0012.JPG", TEST_DATA_DIRECTORY).grayscale_data

    test_image = canon_scan.CanonImage("IMG_0011.JPG", TEST_DATA_DIRECTORY)

    cs.calibrate_flatfield(test_image)

    print(np.mean(test_image.calibrated_image_data))
    test_image.save()

    assert os.path.exists(os.path.join(TEST_DATA_DIRECTORY, "IMG_0011_cal.JPG"))
    low, high = np.quantile(test_image.calibrated_image_data, (0.01, 0.99))
    assert high - low < 10

def test_distortion():
    cs = canon_scan.CalibrationSet()

    cs.flat_field = canon_scan.CanonImage("IMG_0012.JPG", TEST_DATA_DIRECTORY).grayscale_data

    test_image = canon_scan.CanonImage("IMG_0011.JPG", TEST_DATA_DIRECTORY)

    cs.calibrate_flatfield(test_image)

    cs.calibrate_distortion(test_image)

    print(np.mean(test_image.calibrated_image_data))
    test_image.save()

    assert os.path.exists(os.path.join(TEST_DATA_DIRECTORY, "IMG_0011_cal.JPG"))
    low, high = np.quantile(test_image.calibrated_image_data, (0.01, 0.99))
    assert high - low < 10
