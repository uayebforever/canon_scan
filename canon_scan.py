
# Python Standard Library Imports
import os

# Other Library Imports
import numpy as np
from rawkit import raw
from PIL import Image
import cv2
from matplotlib import pyplot



def display_histogram(filename):
    # type: (str) -> np.ndarray

    if filename.endswith(".JPG"):
        with Image.open(filename) as im:
            im_data = np.array(im)

            pyplot.hist(im_data.flat, bins=256)

        return im_data

    with raw.Raw(filename) as rimg:
        rimg.data


class CanonImage:

    def __init__(self, filename, dir=None):
        if dir is not None:
            self.base_dir = dir
            filename = os.path.join(dir, filename)
        else:
            self.base_dir = os.path.dirname(filename)
            # filename = os.path.basename(filename)

        self.filename = os.path.basename(filename)

        self.image = Image.open(filename)

        self.grayscale = self.image.convert("L")

        self._calibrated_image_data = None  # type: np.ndarray

        # Check if there are already outputs that can also be opened
        # (perhaps make this a lazy load to save memory?)
        base_filepath = os.path.splitext(filename)[0]
        if os.path.exists(base_filepath + "_cal.JPG"):
            self.calibrated_image = Image.open(base_filepath + "_cal.JPG")
        else:
            self.calibrated_image = None  # type: Image.Image

    @property
    def grayscale_data(self):
        return np.array(self.grayscale)

    @property
    def calibrated_image_data(self):
        # type: () -> np.ndarray

        # If the image file is available, but the data has not been loaded, load it.
        if self._calibrated_image_data is None and self.calibrated_image:
            self._calibrated_image_data = np.array(self.calibrated_image)

        return self._calibrated_image_data

    @calibrated_image_data.setter
    def calibrated_image_data(self, value):
        self._calibrated_image_data = value
        self.calibrated_image = Image.fromarray(self._calibrated_image_data.astype('uint8'))

    def save(self):

        base_filename = os.path.join(self.base_dir, os.path.splitext(self.filename)[0])

        # Calibrated Image
        self.calibrated_image.save(base_filename + "_cal.JPG")


class CalibrationSet:

    def __init__(self):
        self._flatfield_data = None

        self.calibrated_image_data = None  # type: np.ndarray

    def load_flatfield(self, canon_image):
        # type: (CanonImage) -> None

        self.flat_field = canon_image.grayscale_data

    @property
    def flat_field(self):
        return self._flatfield_data

    @flat_field.setter
    def flat_field(self, value):
        self._flatfield_original_data = value

        self._flatfield_data = value/np.mean(value)

    def calibrate_flatfield(self, image):
        # type: (CanonImage) -> None

        image.calibrated_image_data = image.grayscale_data / self._flatfield_data

    def calibrate_distortion(self, image):

        height, width = image.grayscale_data.shape

        distCoeff = np.zeros((4, 1), np.float64)

        distCoeff[0, 0] = -5.0e-6  # k1 (negative to remove barrel distortion)
        distCoeff[1, 0] = 0.0  # k2
        distCoeff[2, 0] = 0.0  # p1
        distCoeff[3, 0] = 0.0  # p2

        # assume unit matrix for camera
        cam = np.eye(3, dtype=np.float32)

        cam[0, 2] = width / 2.0  # define center x
        cam[1, 2] = height / 2.0  # define center y
        cam[0, 0] = 50.  # define focal length x
        cam[1, 1] = 50.  # define focal length y

        # here the undistortion will be computed
        image.grayscale_data = cv2.undistort(image.grayscale_data, cam, distCoeff)



    def calibrate(self, image):
        # type: (CanonImage) -> None

        self.calibrate_flatfield(image)

        self.calibrate_distortion(image)

# def display_image():
#
#     with raw.Raw() as rimg:
