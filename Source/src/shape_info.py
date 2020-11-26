from dataclasses import dataclass


class ShapeInfo:

    @dataclass
    class PointInfo:
        contour: int
        type: int
        connect_from: int
        connect_to: int

    # index of the first point in contour in the landmark points array
    contour_start_index: list
    # defines if contour is closed (e.g. eye contour) or open (e.g. face contour)
    contour_is_closed: list
    # number of contours in shape
    n_contours: int
    # list of info for every point in shape
    point_info: list()

