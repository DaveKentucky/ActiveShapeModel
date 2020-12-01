from shape_info import ShapeInfo

import numpy as np


class ShapeCreator:

    # list of points
    points: list

    # index of currently marked contour
    current_contour_index: int

    # if currently marked contour is closed or open
    current_is_closed: int

    # index of first point in every contour
    contour_start: list

    # type of every contour
    contour_type: list

    # index of last point in previously marked contour
    end_index: None

    # ShapeInfo object of the model image
    info: ShapeInfo

    def __init__(self, img: np.ndarray):

        self.image = img.copy()
        self.contour_start = list()
        self.contour_type = list()

    def add_point(self, x, y):
        self.points.append(np.array([x, y]))    # add point coordinates to list

        if len(self.contour_start) == 0:
            self.start_contour(0)

    def delete_point(self):
        self.points.pop()

    def start_contour(self, start_index):
        self.contour_start.append(start_index)              # add first index of new contour
        self.contour_type.append(self.current_is_closed)    # add type of the contour
        contour = len(self.contour_start) - 1
        self.current_contour_index = contour

    def create_shape_info(self):
        self.info = ShapeInfo()
        self.info.create_from_shape(self.points, self.contour_start, self.contour_type)
