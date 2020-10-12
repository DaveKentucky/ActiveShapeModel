import numpy as np
import cv2 as cv
import sys


class Image:

    def __init__(self, file, size):

        image = cv.imread(file)
        if image is None:
            sys.exit("Failed to load the image")

        self.image = image
        self.size = size

        self.height = self.image.shape[0]
        self.width = self.image.shape[1]
        self.scale_image(self.width, self.height)

    def scale_image(self, w, h):

        if w == h:
            new_w = new_h = self.size
        elif w > h:
            scale = w/self.size
            new_w = self.size
            new_h = int(h/scale)
        else:
            scale = h/self.size
            new_w = int(w/scale)
            new_h = self.size

        self.image = cv.resize(self.image, (new_w, new_h))
        self.height = self.image.shape[0]
        self.width = self.image.shape[1]
