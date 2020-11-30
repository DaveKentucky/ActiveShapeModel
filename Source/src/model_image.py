from shape_info import ShapeInfo
from shape_vector import ShapeVector

import numpy as np
import cv2 as cv
import sys


class ModelImage:

    # if image was loaded yet
    is_loaded: bool

    # number of landmark points in shape
    n_points: int

    # information about shape
    shape_info: ShapeInfo

    # array of landmark points
    points: np.array

    # shape vectors
    shape_vector: ShapeVector

    # image
    image: np.array

    # image name
    name: str

    def __init__(self):

        self.is_loaded = False
        self.shape_info = ShapeInfo()
        self.shape_vector = ShapeVector()

    def read_from_file(self, directory, file):
        """
        Read training image from given file

        :param directory: track to directory with training images
        :type directory: str
        :param file: path to an image file
        :type file: str
        :return: None
        """
        if not self.is_loaded:
            path = directory + "/" + file
            image = cv.imread(path)
            if image is None:
                sys.exit("Failed to load the image")

            self.image = image
            self.name = file.split('.')[0]
            self.is_loaded = True

    def set_points_from_array(self, p):
        """
        Sets landmark points array with given array of points

        :param p: numpy array of points (Nx2 shape)
        :type p: numpy.ndarray
        :return: None
        """
        self.n_points = p.shape[0]
        self.points = p.copy()
        self.shape_vector.set_from_points_array(p)

    def show(self, show):
        """
        Returns copy of the image with marked model shapes on it

        :param show: if the image should be displayed in a window
        :type show: bool
        :return: image copy with model shapes
        :rtype: numpy.ndarray
        """
        if len(self.image.points) == 1:
            img = cv.cvtColor(self.image, cv.COLOR_GRAY2RGB)
        else:
            img = self.image.copy()

        # TODO: Draw shapes on the image

        if show:
            cv.imshow("Image", img)
            cv.waitKey()

        return img


if __name__ == '__main__':

    mi = ModelImage()
    mi.read_from_file('E:/Szkolne/Praca_inzynierska/ActiveShapeModel/Source/data/Face_images', 'face1.jpg')
    mi.show(True)
