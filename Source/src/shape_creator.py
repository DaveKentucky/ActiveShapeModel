from shape_info import ShapeInfo

import numpy as np
import cv2 as cv


class ShapeCreator:

    # copy of the image being marked
    image: np.ndarray

    # name of the window for marking
    window_name: str

    # list of points
    points: list

    # if currently marked contour is closed or open
    current_is_closed: bool

    # index of first point in every contour
    contour_start: list

    # type of every contour
    contour_type: list

    # index of last point in previously marked contour
    end_index: int

    # ShapeInfo object of the model image
    info: ShapeInfo

    def __init__(self, img, window_name):
        """

        :param img: image
        :type img: numpy.ndarray
        :type window_name: str
        """
        self.image = img.copy()
        self.window_name = window_name
        self.points = list()
        self.current_is_closed = False
        self.contour_start = list()
        self.contour_type = list()
        self.end_index = -1
        self.info = ShapeInfo()

    def __repr__(self):

        string = "points:\n"
        for point in self.points:
            string += str(point) + "\n"

        return f"ShapeCreator: image: {self.window_name},\n" \
               f"{len(self.points)} points and {len(self.contour_start)} contours marked,\n" + string

    def get_display_image(self):

        self.info.create_from_shape(self.points, self.contour_start, self.contour_type)
        image = self.info.draw_points_on_image(self.image, np.array(self.points), False)

        return image

    def add_point(self, x, y):
        """
        Adds new point to the list

        :param x: X coordinate od the point
        :type x: int
        :param y: Y coordinate of the point
        :type y: int
        :return: None
        """
        self.points.append(np.array([x, y]))    # add point coordinates to list

        if self.end_index == len(self.points) - 2:    # it is the first point in new contour
            self.start_contour()

    def delete_point(self):
        """
        Deletes last point in the list

        :return:
        """
        if len(self.points) == 0:
            return
        if self.contour_start[-1] == len(self.points) - 1:  # last point is start of a contour
            self.contour_start.pop()
            self.contour_type.pop()
            if len(self.contour_start) == 0:     # no contours saved currently
                self.end_index = -1
            else:   # return to creating previous contour
                self.end_index = self.points[-1] - 1
        self.points.pop()

    def start_contour(self):
        """
        Starts new contour

        :return: None
        """
        self.contour_start.append(self.end_index + 1)   # add first index of new contour
        self.contour_type.append(int(self.current_is_closed))    # add type of the contour

    def end_contour(self):
        """
        Sets current contour as finished

        :return: None
        """
        self.end_index = len(self.points) - 1

    def flip_contour_type(self):
        """
        Changes type of currently created contour into opposite

        :return: None
        """
        self.current_is_closed = not self.current_is_closed
        if len(self.contour_start) > 0 and len(self.points) - 1 > self.end_index:
            self.contour_type[-1] = int(self.current_is_closed)

    def create_shape_info(self):
        """
        Creates ShapeInfo object from created shape

        :return: None
        """
        self.info.create_from_shape(self.points, self.contour_start, self.contour_type)
        return self.info
