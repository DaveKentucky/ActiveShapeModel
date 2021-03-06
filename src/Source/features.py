from shape_info import ShapeInfo

import cv2 as cv
import numpy as np
import math


class FeatureExtractor:

    # number of levels of the pyramid
    levels: int

    # gaussian pyramid of the image (in greyscale)
    gaussian_pyramid: list

    # laplacian pyramid of the image
    laplacian_pyramid: list

    # info about shapes and contours
    shape_info: ShapeInfo

    # amount of points in each direction on the normal to the shape bounding
    points_per_direction: int

    # amount of points in each direction on the normal to the shape bounding while searching for new points
    search_points_per_direction: int

    def __init__(self, levels, points_per_direction, search_points_per_direction, shape_info=None):
        """
        :param levels: number of levels of the pyramid
        :type levels: int
        :param points_per_direction: amount of points searched along the normal to the shape on each side
        :type points_per_direction: int
        :param search_points_per_direction: amount of points searched along the normal to the shape on each side while
        searching for new points
        :type search_points_per_direction: int
        :type shape_info: ShapeInfo
        """
        self.levels = levels
        self.points_per_direction = points_per_direction
        self.search_points_per_direction = search_points_per_direction
        self.laplacian_pyramid = list()

        if shape_info is not None:
            self.shape_info = shape_info

    def __repr__(self):
        return f"FeatureExtractor: pyramid levels: {self.levels},\n" \
               f"searched points on one side of normal while creating: {self.points_per_direction},\n" \
               f"searched points on one side of normal while searching: {self.search_points_per_direction}"

    def load_image(self, img):
        """
        Loads the image and creates gaussian and laplacian pyramids of it

        :param img: image
        :type img: numpy.ndarray
        :return: None
        """
        layer = img.copy()
        if len(img.shape) == 3:
            layer = cv.cvtColor(layer, cv.COLOR_BGR2GRAY)

        self.gaussian_pyramid = [layer]

        # build layers of gaussian pyramid
        for i in range(self.levels -1):
            layer = cv.pyrDown(layer)
            self.gaussian_pyramid.append(layer)
            # cv.imshow(str(i), layer)

        # build layers of laplacian pyramid
        for gaussian in self.gaussian_pyramid:
            # create higher layer of gaussian pyramid layer
            gaussian_extended = cv.pyrDown(gaussian)
            layer = cv.GaussianBlur(layer, (5, 5), cv.BORDER_DEFAULT)
            # expand the upper layer of gaussian pyramid
            gaussian_extended = cv.pyrUp(gaussian_extended, dstsize=(gaussian.shape[1], gaussian.shape[0]))
            # subtract the layers to get laplacian layer
            laplacian = cv.subtract(gaussian, gaussian_extended)
            laplacian += 100
            self.laplacian_pyramid.append(laplacian)
            # cv.imshow(str(i), laplacian)

    def get_normal_direction(self, points, point_id):
        """
        Computes the direction of normal vector to the shape at given point

        :param points: numpy array of points (Nx2 shape)
        :type points: numpy.ndarray
        :param point_id: index of the point in which the normal vector should be calculated
        :type point_id: int
        :return: direction vector of the normal to the shape (1x2 shape)
        :rtype: numpy.ndarray
        """
        va = points[self.shape_info.point_info[point_id].connect_from] - points[point_id]
        vb = points[self.shape_info.point_info[point_id].connect_to] - points[point_id]

        magnitude = np.linalg.norm(va)
        if magnitude > 1e-10:
            va = va / magnitude
        magnitude = np.linalg.norm(vb)
        if magnitude > 1e-10:
            vb = vb / magnitude

        direction = np.zeros(2)
        direction[0] = - va[1] + vb[1]
        direction[1] = va[0] - vb[0]

        if np.linalg.norm(direction) < 1e-10:
            if np.linalg.norm(va) > 1e-10:
                direction = - va
            else:
                direction = np.array([1, 0])
        else:
            direction = direction / np.linalg.norm(direction)

        return direction

    @staticmethod
    def get_center(loop_range, prev_x, prev_y, direction):
        """
        Finds the center coordinates

        :param loop_range: max range of search iterations
        :type loop_range: int
        :param prev_x: previous x coordinate of the center
        :type prev_x: int
        :param prev_y: previous y coordinate of the center
        :type prev_y: int
        :param direction: direction line parameters
        :type direction: (int, int)
        :return: center coordinates and proper number of iterations
        :rtype: (int, int, int)
        """

        nx = 0
        ny = 0
        j = 1
        for i in range(loop_range):
            while True:
                nx = round(j * direction[0])
                ny = round(j * direction[1])
                j += 1

                if nx != prev_x or ny != prev_y:
                    prev_x = nx
                    prev_y = ny
                    break
            j -= 1

        return nx, ny, j

    def get_points_on_normal(self, points, point_id, level, step, offset=0):
        """
        Finds points on the normal vector to the shape at given point

        :param points: numpy array of points (Nx2 shape)
        :type points: numpy.ndarray
        :param point_id: index of the point in which the normal vector should be calculated
        :type point_id: int
        :param level: analysed level of the gaussian pyramid of the image
        :type level: int
        :param step: step between searched points
        :type step: float
        :param offset: minimal offset of the found points
        :type: float
        :return: array of found point on the normal vector to the shape
        :rtype: numpy.ndarray
        """
        direction = self.get_normal_direction(points, point_id)
        ppd = self.points_per_direction

        nx, ny, j = self.get_center(abs(offset), 0, 0, direction)

        if offset > 0:
            offset_x = nx
            offset_y = ny
        else:
            offset_x = -nx
            offset_y = -ny

        direction *= step
        nx, ny, j = self.get_center(ppd, 0, 0, direction)
        prev_x = nx
        prev_y = ny

        output_points = np.zeros([2 * ppd + 1, 2], np.int)
        for i in range(ppd, -ppd - 1, -1):
            x = int(points[point_id, 0])
            y = int(points[point_id, 1])
            rx = (x >> level) + nx + offset_x
            ry = (y >> level) + ny + offset_y

            if rx < 0:
                rx = 0
            if ry < 0:
                ry = 0
            if rx >= self.gaussian_pyramid[level].shape[1]:
                rx = self.gaussian_pyramid[level].shape[1] - 1
            if ry >= self.gaussian_pyramid[level].shape[0]:
                ry = self.gaussian_pyramid[level].shape[0] - 1

            output_points[i + ppd, 0] = int(rx)
            output_points[i + ppd, 1] = int(ry)

            while True:
                nx = round(j * direction[0])
                ny = round(j * direction[1])
                j -= 1

                if nx != prev_x or ny != prev_y:
                    prev_x = nx
                    prev_y = ny
                    break

        # make sure points' coordinates do not exceed the image size
        for pt in output_points:
            if pt[0] >= self.laplacian_pyramid[level].shape[1]:
                pt[0] = self.laplacian_pyramid[level].shape[1] - 1
            if pt[1] >= self.laplacian_pyramid[level].shape[0]:
                pt[1] = self.laplacian_pyramid[level].shape[0] - 1

        return output_points

    def get_feature(self, points, point_id, level):
        """
        Finds features on image pyramid for the given point

        :param points: array of points
        :type points: numpy.ndarray
        :param point_id: index of the point in array
        :type point_id: int
        :param level: level of the image pyramid to be searched
        :type level: int
        :return: array of features
        :rtype: numpy.ndarray
        """
        x_min = np.min(points[:, 0])
        y_min = np.min(points[:, 1])
        x_max = np.max(points[:, 0])
        y_max = np.max(points[:, 1])
        step = 1.3 * math.sqrt((x_max - x_min) * (y_max - y_min) / 10000.)

        points_on_normal = self.get_points_on_normal(points, point_id, level, step)
        ppd = self.points_per_direction
        array = np.zeros([2 * ppd + 1, 1])

        abs_sum = 0
        for i in range(ppd, -ppd - 1, -1):
            ix = points_on_normal[i + ppd, 0]
            if ix >= self.laplacian_pyramid[level].shape[1]:
                ix = self.laplacian_pyramid[level].shape[1] - 1
            iy = points_on_normal[i + ppd, 1]
            if iy >= self.laplacian_pyramid[level].shape[0]:
                iy = self.laplacian_pyramid[level].shape[0] - 1
            tmp_laplacian = self.laplacian_pyramid[level]
            lap_pt = tmp_laplacian[iy, ix]
            array[i + ppd, 0] = lap_pt
            abs_sum += math.fabs(array[i + ppd, 0])

        if abs_sum != 0:
            array = array / abs_sum

        return array

    def get_candidates_with_feature(self, points, point_id, level):
        """
        Find candidates for new points in a search

        :param points: numpy array of points (Nx2 shape)
        :type points: numpy.ndarray
        :param point_id: index of the point in which the normal vector should be calculated
        :type point_id: int
        :param level: analysed level of the gaussian pyramid of the image
        :type level: int
        :return: vector of found candidate points and vector of features
        :rtype: (list, list)
        """
        candidate_points = list()
        features = list()

        img = self.laplacian_pyramid[level]
        sppd = self.search_points_per_direction
        ppd = self.points_per_direction

        for i in range(-sppd, sppd + 1, 1):
            points_on_normal = self.get_points_on_normal(points, point_id, level, 1, i)

            normals_vector = np.zeros([2 * ppd + 1])
            abs_sum = 0
            for j in range(-ppd, ppd + 1, 1):
                pt_x = int(points_on_normal[j + ppd][0])
                pt_y = int(points_on_normal[j + ppd][1])
                normals_vector[j + ppd] = img[pt_y, pt_x]
                abs_sum += math.fabs(normals_vector[j + ppd])

            if abs_sum > 0:
                normals_vector = normals_vector / abs_sum
            candidate_points.append(points_on_normal[ppd])
            features.append(normals_vector)

        return candidate_points, features


if __name__ == '__main__':
    image = cv.imread("E:/Szkolne/Praca_inzynierska/ActiveShapeModel/src/Data/face_database/01-1m.jpg")
    # cv.imshow("Original image", image)
    fe = FeatureExtractor(3, 5, 5)
    fe.load_image(image)
    for i in range(len(fe.gaussian_pyramid)):
        cv.imshow("gaussian", fe.gaussian_pyramid[i])
        cv.imshow("laplacian", fe.laplacian_pyramid[i])
        cv.waitKey(0)
        cv.destroyWindow("gaussian")
        cv.destroyWindow("laplacian")
    # cv.waitKey(0)
    # cv.destroyAllWindows()

