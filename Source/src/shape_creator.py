from shape_info import ShapeInfo

import numpy as np


class ShapeCreator:

    # list of points
    points: list

    # index of currently marked contour
    current_contour_index: int

    # if currently marked contour is closed or open
    current_is_closed: int

    # index of first point in currently marked contour
    start_index: int

    # index of last point in previously marked contour
    end_index: None

    # ShapeInfo object of the model image
    info: ShapeInfo

    def __init__(self, img: np.ndarray):

        self.image = img.copy()
        self.info = ShapeInfo()

    def add_point(self, x, y):
        self.points.append(np.array([x, y]))    # add point coordinates to list

        if self.info.n_contours == 0:   # it is the first contour in this shape
            self.info.add_contour(0, self.current_is_closed)
            self.current_contour_index = 0
            self.start_index = 0
        if len(self.points) - 1 == self.start_index:    # it is start point of a new contour
            self.info.add_point_info(True)
        else:   # it is another point in this contour
            self.info.add_point_info(False)

    def delete_point(self):
        self.points.pop()

        self.info.delete_last_point_info()
