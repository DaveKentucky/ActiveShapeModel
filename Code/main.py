import sys
import numpy as np
import cv2 as cv
from image import Image


def mouse_click(event, x, y, flags, param):
    # resolve mouse click on image

    # left button clicked
    if event == cv.EVENT_LBUTTONDOWN:
        point = (x, y)
        # print(point)

        # add landmark point to image's list
        img.add_landmark_point(x, y)
        cv.imshow("Image", img.get_image_with_landmarks())

filename = "img6"
img = Image("Images/Face/" + filename + ".jpg", 400)

print("height: " + str(img.height))
print("width: " + str(img.width))

cv.namedWindow("Image", cv.WINDOW_KEEPRATIO)
cv.setMouseCallback("Image", mouse_click, img)
cv.imshow("Image", img.get_image_with_landmarks())

wait_time = 1000
while cv.getWindowProperty("Image", cv.WND_PROP_VISIBLE) >= 1:
    key = cv.waitKey(wait_time)
    if key == 27:
        cv.destroyWindow("Image")
    elif key == ord('r'):
        img.remove_landmark_point()
        cv.imshow("Image", img.get_image_with_landmarks())
