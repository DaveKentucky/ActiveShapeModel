import sys
import numpy as np
import cv2 as cv
from image import Image


def mouse_click(event, x, y, flags, param):
    # resolve mouse click on image

    if event == cv.EVENT_LBUTTONDOWN:
        point = (x, y)
        # print(point)
        img.add_landmark_point(x, y)
        cv.imshow("Image", img.image)

filename = "img6"
img = Image("Images/Face/" + filename + ".jpg", 400)

print("height: " + str(img.height))
print("width: " + str(img.width))

cv.namedWindow("Image")
cv.setMouseCallback("Image", mouse_click, img)

while True:
    cv.imshow("Image", img.image)
    key = cv.waitKey(0)
    if key == 27:
        break
