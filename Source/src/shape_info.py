from dataclasses import dataclass
import cv2 as cv


class ShapeInfo:

    # dataclass of single point information
    @dataclass
    class PointInfo:
        contour: int        # index of contour (contour_start_index)
        type: int           # closed (1) or open (0)
        connect_from: int   # index of previous point
        connect_to: int     # index of next point

    # index of the first point in contour in the landmark points array
    contour_start_index: list

    # defines if contour is closed (e.g. eye contour) or open (e.g. face contour)
    contour_is_closed: list

    # number of contours in shape
    n_contours: int

    # list of info for every point in shape
    point_info: list

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
