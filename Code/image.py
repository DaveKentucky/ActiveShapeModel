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

        self._image = image  # jpeg image
        self.size = size    # desired max size of image
        self.points = []    # array for landmark points

        # set original size
        self.height = self._image.shape[0]
        self.width = self._image.shape[1]
        # scale the image ot desired size
        self.scale(self.width, self.height)

    def scale(self, w, h):
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

        self._image = cv.resize(self._image, (new_w, new_h))
        self.height = self._image.shape[0]
        self.width = self._image.shape[1]

    def add_landmark_point(self, x, y):
        # add new point to the list of landmark points
        # and draw it on the displayed image

        self.points.append((x, y))

        print("landmark points marked:")
        for point in self.points:
            print(point)

    def remove_landmark_point(self):
        # remove the last landmark point from the list

        if len(self.points) > 0:
            del self.points[-1]
            print("landmark point deleted")
            if len(self.points) > 0:
                print("landmark points marked:")
                for point in self.points:
                    print(point)
        else:
            print("no landmark points marked \n")

    def get_display_image(self):
        # get image with notified landmark points on it

        display_image = self._image.copy()

        for point in self.points:
            cv.circle(display_image, (point[0], point[1]), 3, (0, 0, 255), -1)

        return display_image
