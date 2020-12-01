from dataclasses import dataclass
import cv2 as cv


# dataclass of single point information
@dataclass
class PointInfo:
    contour: int  # index of contour in contour_start_index list
    type: int  # closed (1) or open (0)
    connect_from: int  # index of previous point
    connect_to: int  # index of next point


class ShapeInfo:

    # index of the first point in contour in the landmark points array
    contour_start_index: list

    # defines if contour is closed (e.g. eye contour) or open (e.g. face contour)
    contour_is_closed: list

    # number of contours in shape
    n_contours: int

    # list of info for every point in shape
    point_info: list

    def __init__(self):
        self.n_contours = 0

    def __repr__(self):
        return f"ShapeInfo: {self.n_contours} contours, start indexes: {self.contour_start_index}"

    def create_from_shape(self, points, start_indices, types):
        """
        Creates ShapeInfo object from lists describing shape

        :param points: list of points
        :type points: list[int, int]
        :param start_indices: list of start indices of every contour
        :type start_indices: list[int]
        :param types: list of types of every contour
        :type types: list[int]
        :return: None
        """
        self.n_contours = len(start_indices)
        self.contour_start_index = start_indices.copy()
        self.contour_is_closed = types.copy()
        self.point_info = list()

        for i, contour_start in enumerate(self.contour_start_index):
            for j, point in enumerate(points):
                max_index = self.contour_start_index[i + 1] if i + 1 < len(self.contour_start_index) else len(points)
                if contour_start <= j < max_index:   # every point in this contour
                    if j == contour_start:  # first point in contour
                        if self.contour_is_closed[i] == 1:  # first point in closed contour
                            c_from = max_index - 1
                        else:   # first point in open contour
                            c_from = j
                    else:   # connect to previous point otherwise
                        c_from = j - 1
                    if j == max_index - 1:    # last point in contour
                        if self.contour_is_closed[i] == 1:  # last point in closed contour
                            c_to = contour_start
                        else:   # last point in open contour
                            c_to = j
                    else:   # connect to next point otherwise
                        c_to = j + 1

                    p_info = PointInfo(i, self.contour_is_closed[i], c_from, c_to)
                    self.point_info.append(p_info)

    def draw_points_on_image(self, image, points, draw_directly):
        """
        Puts points and contour lines on given image

        :param image: canvas to draw points on
        :type image: numpy.ndarray
        :param points: array of points (Nx2 shape)
        :type points: numpy.ndarray
        :param draw_directly: if points should be drawn on this image or its copy
        :type draw_directly: bool
        :return: image with drawn points and contours
        :rtype: numpy.ndarray
        """
        if draw_directly:
            img = image
        else:
            img = image.copy()

        for point in points:
            cv.circle(img, (point[0], point[1]), 3, (0, 0, 220), -1)

        for i in range(self.n_contours):
            for j, point in enumerate(points):
                if j < self.contour_start_index[i + 1]:
                    cv.line(img, points[self.point_info[j].connect_from], point, (50, 100, 200), 1)

        return img
