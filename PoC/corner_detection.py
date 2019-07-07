import cv2
import numpy as np

filename = 'IMG_0014_cal.JPG'
img = cv2.imread(filename)
down_sampled_img = cv2.resize(img, None, fx=0.1, fy=0.1)
gray = cv2.cvtColor(down_sampled_img, cv2.COLOR_BGR2GRAY)

gray = np.float32(gray)
dst = cv2.cornerHarris(gray, 2, 3, 0.04)

#result is dilated for marking the corners, not important
dst = cv2.dilate(dst,None)

# Threshold for an optimal value, it may vary depending on the image.
down_sampled_img[dst > 0.01 * dst.max()]=[0, 0, 255]

cv2.imshow('dst', down_sampled_img)
if cv2.waitKey(0) & 0xff == 27:
    cv2.destroyAllWindows()