import numpy as np
import cv2
import glob

# termination criteria
criteria = (cv2.TERM_CRITERIA_EPS + cv2.TERM_CRITERIA_MAX_ITER, 30, 0.001)

chess_board_rows = 7
chess_board_cols = 9

# prepare object points, like (0,0,0), (1,0,0), (2,0,0) ....,(6,5,0)
objp = np.zeros((chess_board_rows * chess_board_cols, 3), np.float32)
objp[:, :2] = np.mgrid[0:chess_board_rows, 0:chess_board_cols].T.reshape(-1, 2)

# Arrays to store object points and image points from all the images.
objpoints = []  # 3d point in real world space
imgpoints = []  # 2d points in image plane.

images = map("IMG_{0:04d}.JPG".format, range(25, 42))

gray = None

for fname in images:
    print("Proccessing image %s" % fname)
    img = cv2.imread(fname)
    gray = cv2.cvtColor(img, cv2.COLOR_BGR2GRAY)

    # Find the chess board corners
    ret, corners = cv2.findChessboardCorners(gray, (chess_board_rows, chess_board_cols), None)

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

assert gray is not None

ret, mtx, dist, rvecs, tvecs = cv2.calibrateCamera(objpoints, imgpoints, gray.shape[::-1], None, None)

img = cv2.imread('IMG_0035.JPG')
h, w = img.shape[:2]
newcameramtx, roi = cv2.getOptimalNewCameraMatrix(mtx, dist, imageSize=(w, h), alpha=0, newImgSize=(w, h))

# undistort
mapx, mapy = cv2.initUndistortRectifyMap(mtx, dist, None, newcameramtx, (w, h), 5)
dst = cv2.remap(img, mapx, mapy, cv2.INTER_LINEAR)

# crop the image
x, y, w, h = roi
dst = dst[y:y + h, x:x + w]
cv2.imwrite('calibresult.png', dst)

tot_error = 0

mean_error = 0
for i in range(len(objpoints)):
    imgpoints2, _ = cv2.projectPoints(objpoints[i], rvecs[i], tvecs[i], mtx, dist)
    error = cv2.norm(imgpoints[i],imgpoints2, cv2.NORM_L2)/len(imgpoints2)
    tot_error += error

print("total error: ", mean_error/len(objpoints))


# Save calibration data to file:
np.savez("camera_calibration.npz", mapx=mapx, mapy=mapy, dist=dist, newcameramtx=newcameramtx, mtx=mtx)


# Perspective correction

img = cv2.imread("IMG_0046.JPG")

dst = cv2.remap(img, mapx, mapy, cv2.INTER_LINEAR)

# crop the image
x, y, w, h = roi
dst = dst[y:y + h, x:x + w]
cv2.imwrite('IMG_0046_dist.png', dst)

gray = cv2.cvtColor(img, cv2.COLOR_BGR2GRAY)

# Find the chess board corners
ret, corners = cv2.findChessboardCorners(gray, (chess_board_rows, chess_board_cols), None)
# corners2 = cv2.cornerSubPix(gray, corners, (11, 11), (-1, -1), criteria)

# Draw and display the corners
# img = cv2.drawChessboardCorners(img, (7, 6), corners, ret)
# cv2.imshow('img', img)
# cv2.waitKey(0)

# cv2.destroyAllWindows()

# Find corners of grid
#
#  NOTE: This needs work, as it can easily fail.

# corner_indexes = [corners[:,0,0].argmin(), corners[:,0,0].argmax(), corners[:,0,1].argmin(), corners[:,0,1].argmax()]
# assert len(np.unique(np.array(corner_indexes))) == 4
height, width = dst.shape[:2]
corner_indexes = [np.linalg.norm(corners, axis=2).argmin(), # Bottom Left
                  np.linalg.norm(corners, axis=2).argmax(), # Top Right
                  np.linalg.norm(corners + np.array([[[-width,0]]]), axis=2).argmin(),  # Bottom Right
                  np.linalg.norm(corners + np.array([[[-width,0]]]), axis=2).argmax()]  # Top left
assert len(np.unique(np.array(corner_indexes))) == 4

norm = np.linalg.norm

bl, tr, br, tl = corners[corner_indexes,0]

chess_board_ratio = (chess_board_cols - 1) / (chess_board_rows - 1)

# Take two left corners, and compute new right corners.
offset = np.array([chess_board_ratio * ((tl - bl)[1]), -chess_board_ratio * ((tl - bl)[0])])
new_br = bl + offset
new_tr = tl + offset

perspective_transform = cv2.getPerspectiveTransform(corners[corner_indexes, 0, :],
                                                    np.array([bl, new_tr, new_br, tl], dtype=np.float32))

final_img = cv2.warpPerspective(dst, perspective_transform, (width, height))

cv2.imwrite('IMG_0046_final.png', final_img)

