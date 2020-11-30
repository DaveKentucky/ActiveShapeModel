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

    def draw_points_on_image(self, image, points, draw_directly):

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

    def add_contour(self, start_id, is_closed):
        self.contour_start_index.append(start_id)   # set index of first point in this contour
        self.contour_is_closed.append(is_closed)    # set if this contour is closed or open
        self.n_contours += 1
        self.add_point_info(True)

    def add_point_info(self, start, contour_id=None):
        if contour_id is None:  # on default add to the last contour
            contour_id = len(self.contour_start_index) - 1
        is_closed = self.contour_is_closed[contour_id]  # check if contour is closed or open
        if start:   # it is the first point in this contour
            p_info = PointInfo(contour_id, is_closed, len(self.point_info), len(self.point_info))
        else:       # it is another point in this contour
            p_info = PointInfo(contour_id, is_closed, len(self.point_info) - 1, len(self.point_info))
        self.point_info.append(p_info)

    def delete_last_point_info(self):
        last_id = len(self.point_info) - 1
        self.point_info.pop()
        if last_id in self.contour_start_index:     # deleted point was first in a contour
            self.contour_start_index.pop()
            self.contour_is_closed.pop()
        else:   # deleted point was part of a contour
            self.point_info[-1].connect_to = last_id - 1
