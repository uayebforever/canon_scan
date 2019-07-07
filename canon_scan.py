
from typing import List, Dict

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

            pyplot.hist(im_data.flat, bins=range(0, 255))

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
        if value is not None:
            self._calibrated_image_data = value
            self.calibrated_image = Image.fromarray(self._calibrated_image_data.astype('uint8'))

    def save(self):

        base_filename = os.path.join(self.base_dir, os.path.splitext(self.filename)[0])

        # Calibrated Image
        self.calibrated_image.save(base_filename + "_cal.JPG")


class CalibrationSet:

    chess_board_rows = 7
    chess_board_cols = 9

    def __init__(self):
        self._flatfield_data = None

        self.calibrated_image_data = None  # type: np.ndarray

        self.distortion_measurement_filenames = []  # type: List[str]
        self.distortion_measurement_data = dict()  # type: Dict[str, np.ndarray]

        self.perspective_transform = None

        self._undistort_maps = None

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


    def measure_distortion(self, save="camera_distortion_model.npz"):
        # termination criteria
        criteria = (cv2.TERM_CRITERIA_EPS + cv2.TERM_CRITERIA_MAX_ITER, 30, 0.001)

        # prepare object points, like (0,0,0), (1,0,0), (2,0,0) ....,(6,5,0)
        objp = np.zeros((self.chess_board_rows * self.chess_board_cols, 3), np.float32)
        objp[:, :2] = np.mgrid[0:self.chess_board_rows, 0:self.chess_board_cols].T.reshape(-1, 2)

        # Arrays to store object points and image points from all the images.
        objpoints = []  # 3d point in real world space
        imgpoints = []  # 2d points in image plane.

        images = map("IMG_{0:04d}.JPG".format, range(25, 42))

        gray = None
        img = None

        for fname in self.distortion_measurement_filenames:
            print("Proccessing image %s" % fname)
            img = cv2.imread(fname)
            gray = cv2.cvtColor(img, cv2.COLOR_BGR2GRAY)

            # Find the chess board corners
            ret, corners = cv2.findChessboardCorners(gray, (self.chess_board_rows, self.chess_board_cols), None)

            # If found, add object points, image points (after refining them)
            if ret:
                print("   found %s points" % str(len(objp)))
                objpoints.append(objp)

                corners2 = cv2.cornerSubPix(gray, corners, (11, 11), (-1, -1), criteria)
                imgpoints.append(corners2)

                # Draw and display the corners
                # img = cv2.drawChessboardCorners(img, (7, 6), corners2, ret)
                # cv2.imshow('img', img)
                # cv2.waitKey(500)
            else:
                print("   no points found.")

        # cv2.destroyAllWindows()

        assert isinstance(gray, np.ndarray), isinstance(img, np.ndarray)

        ret, mtx, dist, rvecs, tvecs = cv2.calibrateCamera(objpoints, imgpoints, gray.shape[::-1], None, None)

        # img = cv2.imread('IMG_0035.JPG')
        h, w = img.shape[:2]
        newcameramtx, roi = cv2.getOptimalNewCameraMatrix(mtx, dist, imageSize=(w, h), alpha=0, newImgSize=(w, h))

        self.distortion_measurement_data = {
            'input_camera_matrix': mtx,
            'distortion_coefficients': dist,
            'new_camera_matrix': newcameramtx,
            'roi': roi
        }
        if save:
            np.savez(save, **self.distortion_measurement_data)

    def load_distortion_measurement_data(self, filename):
        # type: (str) -> None

        self.distortion_measurement_data = np.load(filename)


    def measure_perspecitve(self, image):
        # type: (CanonImage) -> None

        # Correct for distortion:

        dst = self.calibrate_distortion(image, 'grayscale_data')

        # Get image size
        height, width = dst.shape[:2]

        # Find the chess board corners
        ret, corners = cv2.findChessboardCorners(dst, (self.chess_board_rows, self.chess_board_cols), None)
        # corners2 = cv2.cornerSubPix(gray, corners, (11, 11), (-1, -1), criteria)

        # Find corners of grid
        corner_indexes = [np.linalg.norm(corners, axis=2).argmin(),  # Bottom Left
                          np.linalg.norm(corners, axis=2).argmax(),  # Top Right
                          np.linalg.norm(corners + np.array([[[-width, 0]]]), axis=2).argmin(),  # Bottom Right
                          np.linalg.norm(corners + np.array([[[-width, 0]]]), axis=2).argmax()]  # Top left
        assert len(np.unique(np.array(corner_indexes))) == 4

        bl, tr, br, tl = corners[corner_indexes, 0]

        chess_board_ratio = (self.chess_board_cols - 1) / (self.chess_board_rows - 1)

        # Take two left corners, and compute new right corners.
        offset = np.array([chess_board_ratio * ((tl - bl)[1]), -chess_board_ratio * ((tl - bl)[0])])
        new_br = bl + offset
        new_tr = tl + offset

        self.perspective_transform = cv2.getPerspectiveTransform(corners[corner_indexes, 0, :],
                                                            np.array([bl, new_tr, new_br, tl], dtype=np.float32))



    def calibrate_flatfield(self, image, source='grayscale_data'):
        # type: (CanonImage, str) -> np.ndarray

        source_data = getattr(image, source)

        return source_data / self._flatfield_data

    def calibrate_distortion(self, image, source='calibrated_image_data', method='simple'):
        # type: (CanonImage, str, str) -> np.ndarray

        source_data = getattr(image, source)

        height, width = source_data.shape

        if method == "calculate":

            # Calculate and cache un-distortion maps
            if self._undistort_maps is None:
                self._undistort_maps = cv2.initUndistortRectifyMap(
                    self.distortion_measurement_data['input_camera_matrix'],
                    self.distortion_measurement_data['distortion_coefficients'],
                    None,
                    self.distortion_measurement_data['new_camera_matrix'], (width, height), 5)

            # Remap the image data
            undist_image_data = cv2.remap(source_data, self._undistort_maps[0], self._undistort_maps[1], cv2.INTER_LINEAR)

            # crop the image
            x, y, w, h = self.distortion_measurement_data['roi']
            undist_image_data = undist_image_data[y:y + h, x:x + w]

            return undist_image_data

        else:
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
            return cv2.undistort(source_data, cam, distCoeff)


    def calibrate_perspective(self, image, source='calibrated_image_data'):
        # type: (CanonImage, str) -> np.ndarray

        source_data = getattr(image, source)

        height, width = source_data.shape

        return cv2.warpPerspective(source_data, self.perspective_transform, (width, height))



    def calibrate(self, image):
        # type: (CanonImage) -> None

        # Copy the uncalibrated data into the calibrated data, and then run
        # the various calibrations. Each calibration will run on the data
        # in the calibrated attribute, progressing the calibration.

        # image.calibrated_image_data = image.grayscale_data

        image.calibrated_image_data = self.calibrate_flatfield(image)

        image.calibrated_image_data = self.calibrate_distortion(image, method='calculate')

        image.calibrated_image_data = self.calibrate_perspective(image)

# def display_image():
#
#     with raw.Raw() as rimg:
