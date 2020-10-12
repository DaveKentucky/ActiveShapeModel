import numpy as np
import cv2 as cv
import sys


class Image:

    def __init__(self, file, size):
        # file - relative track to image file
        # size - desired size of image (longer edge)

        # read image from file
        image = cv.imread(file)
        if image is None:
            sys.exit("Failed to load the image")

        self.image = image  # jpeg image
        self.size = size    # desired max size of image
        self.points = []    # array for landmark points

        self.height = self.image.shape[0]
        self.width = self.image.shape[1]
        # scale the image ot desired size
        self.scale_image(self.width, self.height)

    def scale_image(self, w, h):
        # w - original image width in pixels
        # h - original image height in pixels

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

    def add_landmark_point(self, x, y):

        self.points.append((x, y))
        cv.circle(self.image, (x, y), 3, (0, 0, 255), -1)

        for point in self.points:
            print(point)
