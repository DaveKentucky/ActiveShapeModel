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

    def draw_points_on_image(self, image, points, draw_directly, labels=True):
        """
        Puts points and contour lines on given image

        :param image: canvas to draw points on
        :type image: numpy.ndarray
        :param points: array of points (Nx2 shape)
        :type points: numpy.ndarray
        :param draw_directly: if points should be drawn on this image or its copy
        :type draw_directly: bool
        :param labels: if numbers of points should be drawn on the image
        :type labels: bool
        :return: image with drawn points and contours
        :rtype: numpy.ndarray
        """
        if draw_directly:
            img = image
        else:
            img = image.copy()

        line_thickness = 2
        point_size = 3
        # draw marked points on image
        for i, point in enumerate(points):
            cv.circle(img, (point[0], point[1]), point_size, (0, 0, 220), -1)
            if labels:
                cv.putText(img, str(i + 1), (point[0] + 2, point[1] - 2), cv.QT_FONT_BLACK, 0.7, (0, 0, 220))

        for i, first in enumerate(self.contour_start_index):  # loop every contour
            for j, point in enumerate(points):  # loop every point
                if i + 1 < len(self.contour_start_index):  # it is NOT the last contour
                    if first <= j:  # point index is larger than first in this contour
                        if j < self.contour_start_index[i + 1]:  # point index is smaller than first in next contour
                            if j == first:  # first point in contour
                                draw_line = False
                            else:  # NOT first point in contour
                                draw_line = True
                        else:  # point DOES NOT belong to current contour
                            draw_line = False
                    else:  # point index is smaller than first in this contour
                        draw_line = False
                else:  # it is the last contour
                    if first <= j:  # point index is larger than first in this contour
                        if j == first:  # first point in contour
                            draw_line = False
                        else:  # NOT first point in contour
                            draw_line = True
                    else:  # point index is smaller than first in this contour
                        draw_line = False
                if draw_line:  # draw line for every point except first points in every shape
                    cv.line(img, tuple(points[j - 1]), tuple(point), (200, 100, 50), line_thickness)

            if i + 1 < len(self.contour_start_index):  # it is NOT the last contour
                if self.contour_is_closed[i] == 1:  # it is a closed contour
                    cv.line(img, tuple(points[first]), tuple(points[self.contour_start_index[i + 1] - 1]),
                            (200, 100, 50), line_thickness)
            else:
                if self.contour_is_closed[i] == 1:  # it is a closed contour
                    cv.line(img, tuple(points[first]), tuple(points[-1]), (200, 100, 50), line_thickness)

        return img
