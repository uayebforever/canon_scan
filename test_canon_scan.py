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

    test_image.calibrated_image_data = cs.calibrate_flatfield(test_image)

    print(np.mean(test_image.calibrated_image_data))
    test_image.save()

    assert os.path.exists(os.path.join(TEST_DATA_DIRECTORY, "IMG_0011_cal.JPG"))
    low, high = np.quantile(test_image.calibrated_image_data, (0.01, 0.99))
    assert high - low < 10

def test_distortion():
    cs = canon_scan.CalibrationSet()

    cs.flat_field = canon_scan.CanonImage("IMG_0012.JPG", TEST_DATA_DIRECTORY).grayscale_data

    test_image = canon_scan.CanonImage("IMG_0014.JPG", TEST_DATA_DIRECTORY)

    test_image.calibrated_image_data = cs.calibrate_flatfield(test_image)

    test_image.calibrated_image_data = cs.calibrate_distortion(test_image)

    print(np.mean(test_image.calibrated_image_data))
    test_image.save()

    assert os.path.exists(os.path.join(TEST_DATA_DIRECTORY, "IMG_0011_cal.JPG"))

def test_measure_distortion():
    format_str = os.path.join(TEST_DATA_DIRECTORY, "IMG_{0:04d}.JPG")
    images = map(format_str.format, range(25, 42))

    cs = canon_scan.CalibrationSet()

    cs.distortion_measurement_filenames = images

    cs.measure_distortion(save=os.path.join(TEST_DATA_DIRECTORY, "camera_distortion_model.npz"))

    print(cs.distortion_measurement_data)
    for name in ('input_camera_matrix', 'distortion_coefficients', 'new_camera_matrix'):
        assert name in cs.distortion_measurement_data

def test_complete_calibration():

    cs = canon_scan.CalibrationSet()

    cs.flat_field = canon_scan.CanonImage("IMG_0051.JPG", TEST_DATA_DIRECTORY).grayscale_data

    cs.load_distortion_measurement_data(os.path.join(TEST_DATA_DIRECTORY, "camera_distortion_model.npz"))

    cs.measure_perspecitve(canon_scan.CanonImage("IMG_0053.JPG", TEST_DATA_DIRECTORY))

    test_img = canon_scan.CanonImage("IMG_0058.JPG", TEST_DATA_DIRECTORY)

    cs.calibrate(test_img)

    test_img.save()