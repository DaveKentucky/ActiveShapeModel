import sys
import numpy as np
import cv2 as cv
from image import Image


filename = "img2"
img = Image("Images/Face/" + filename + ".jpg", 400)

print("height: " + str(img.height))
print("width: " + str(img.width))

cv.imshow("Image", img.image)
key = cv.waitKey(0)