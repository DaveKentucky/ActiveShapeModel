import procrustes_analysis as pa

import numpy as np
import cv2 as cv
import math


class ShapeVector:

    # vector of points
    vector: np.array

    # number of points in vector
    n_points: int

    def __init__(self):

        self.vector = np.array([])
        self.n_points = 0

    def set_from_points_array(self, p):
        """
        Sets vector with array of points

        :param p: numpy array of points (Nx2 shape)
        :type p: numpy.ndarray
        """
        self.n_points = p.shape[0]
        array = np.zeros([2 * self.n_points], np.float)
        for i, point in enumerate(p):
            array[2 * i] = point[0]
            array[2 * i + 1] = point[1]

        self.vector = array

    def set_from_vector(self, v):
        """
        Sets vector with copy of given vector

        :param v: vector of points (2Nx1 array)
        :type v: numpy.ndarray
        :return: None
        """
        self.n_points = int(len(v) / 2)
        self.vector = v.copy()

    def get_point(self, i):
        """
        Returns coordinates of point at given index

        :param i: index of the point
        :return: tuple of point coordinates (X, Y)
        :rtype: (float, float)
        """
        return self.vector[2 * i], self.vector[2 * i + 1]

    def get_x_mean(self):
        """
        :return: mean of the points' X coordinates
        :rtype: float
        """
        return np.mean(self.vector[0::2])

    def get_y_mean(self):
        """
        :return: mean of the points' Y coordinates
        :rtype: float
        """
        return np.mean(self.vector[1::2])

    def to_points_array(self):
        """
        Returns array of points

        :return: array of points (Nx2 shape)
        :rtype: numpy.ndarray
        """
        array = np.zeros((self.n_points, 2), np.float)
        for i in range(self.n_points):
            array[i, 0] = self.vector[2 * i]
            array[i, 1] = self.vector[2 * i + 1]

        return array

    def translate(self, vx, vy):
        """
        Translates the vector with given translation values

        :param vx: X translation
        :type vx: float
        :param vy: Y translation
        :type vy: float
        :return: None
        """
        for i in range(self.n_points):
            self.vector[2 * i] += vx
        for i in range(self.n_points):
            self.vector[2 * i + 1] += vy

    def scale(self, s):
        """
        Scale the vector with given scale

        :param s: scale
        :type s: float
        :return: None
        """
        self.vector = np.multiply(self.vector, s)

    def scale_to_one(self):
        """
        Scale the vector to values in range [0.0, 1.0]

        :return: None
        """
        self.scale(1 / np.linalg.norm(self.vector, np.inf))

    def move_to_origin(self):
        """
        Translates the vector's center to origin

        :return: None
        """
        self.translate(-self.get_x_mean(), -self.get_y_mean())

    def align_to(self, ref_vec):
        """
        Aligns the vector to given vector at origin

        :param ref_vec: reference shape vector object
        :type ref_vec: ShapeVector
        :return: None
        """
        self.vector = pa.procrustes_analysis(ref_vec.vector, self.vector)

    def add_vector(self, vec):
        """
        Sums the vector with given vector

        :param vec: component vector
        :type vec: ShapeVector
        :return: None
        """
        if self.n_points == vec.n_points:
            for i, c in enumerate(vec.vector):
                self.vector[i] += c

    def subtract_vector(self, vec):
        """
        Subtracts given vector from the vector

        :param vec: dividend vector
        :type vec: ShapeVector
        :return: None
        """
        if self.n_points == vec.n_points:
            for i, c in enumerate(vec.vector):
                self.vector[i] -= c

    def restore_to_point_list(self, sim_trans):
        """
        Perform a similarity transformation and restore vector to list of points

        :param sim_trans: transformation object of the shape
        :type sim_trans: SimilarityTransformation
        :return: array of points (Nx2 array)
        :rtype: numpy.ndarray
        """
        pts_vec = np.zeros([self.n_points, 2], np.int)
        sv = ShapeVector()
        sv = sim_trans.transform(self, sv)

        for i in range(sv.n_points):
            pts_vec[i, 0] = sv.vector[i * 2]
            pts_vec[i, 1] = sv.vector[i * 2 + 1]

        return pts_vec

    def get_bound_rectangle(self):
        """
        Finds rectangle bounding the shape

        :return: coordinates of top left point of the rectangle and its shape as tuple: (width, height)
        :rtype: ((float, float), (float, float))
        """
        x_min = np.min(self.vector[::2])
        x_max = np.max(self.vector[::2])
        y_min = np.min(self.vector[1::2])
        y_max = np.max(self.vector[1::2])

        return (x_min, y_min), (x_max - x_min, y_max - y_min)

    def get_shape_transform_fitting_size(self, size, scale_ratio=0.9, offset_x=0, offset_y=0):
        """
        Finds the proper transformation to rescale the shape to given size

        :param size: target size for the shape as tuple: (height, width)
        :type size: (float, float)
        :param scale_ratio: ratio of the target scale
        :type scale_ratio: float
        :param offset_x: horizontal offset of the shape in the image
        :type offset_x: int
        :param offset_y: vertical offset of the shape in the image
        :type offset_y: int
        :return: fitting similarity transformation
        :rtype: SimilarityTransformation
        """
        bound_corner, bound_size = self.get_bound_rectangle()
        ratio_x = size[1] / bound_size[0]
        ratio_y = size[0] / bound_size[1]
        if ratio_x < ratio_y:
            ratio = ratio_x
        else:
            ratio = ratio_y
        ratio *= scale_ratio

        st = SimilarityTransformation()
        trans_x = bound_corner[0] - bound_size[0] * (ratio_x / ratio - 1 + offset_x) / 2
        trans_y = bound_corner[1] - bound_size[1] * (ratio_y / ratio - 1 + offset_y) / 2
        st.a = ratio
        st.b = 0
        st.x_t = -trans_x * ratio
        st.y_t = -trans_y * ratio

        return st


class SimilarityTransformation:

    # X transformation
    x_t: float

    # Y transformation
    y_t: float

    # a in similarity transformation matrix
    a: float

    # b in similarity transformation matrix
    b: float

    def __init__(self):
        self.x_t = 0
        self.y_t = 0
        self.a = 1
        self.b = 0

    def __repr__(self):
        return f"a: {self.a}, b: {self.b}, x_t: {self.x_t}, y_t: {self.y_t}"

    def get_scale(self):
        return math.sqrt(self.a * self.a + self.b * self.b)

    def multiply(self, trans):
        """
        Multiplies two transformations

        :param trans: other transformation
        :type trans: SimilarityTransformation
        :return: result transformation
        :rtype: SimilarityTransformation
        """
        new_trans = SimilarityTransformation()
        new_trans.a = self.a * trans.a - self.b * trans.b
        new_trans.b = self.a * trans.b + self.b * trans.a
        new_trans.x_t = self.a * trans.x_t - self.b * trans.y_t + self.x_t
        new_trans.y_t = self.b * trans.x_t + self.a * trans.y_t + self.y_t

        return new_trans

    def transform(self, vec_src, vec_dst):
        """
        :param vec_src:
        :type vec_src: ShapeVector
        :param vec_dst:
        :type vec_dst: ShapeVector
        :return:
        :rtype: ShapeVector
        """
        n_points = vec_src.n_points
        dst = np.zeros([n_points, 2])

        for i in range(n_points):
            x, y = vec_src.get_point(i)
            dst[i, 0] = self.a * x - self.b * y + self.x_t
            dst[i, 1] = self.b * x + self.a * y + self.y_t
        vec_dst.set_from_points_array(dst)

        return vec_dst

    def inverted_transform(self, vec_src):
        """
        :param vec_src:
        :type vec_src: ShapeVector
        :return:
        :rtype: ShapeVector
        """
        vec_dst = ShapeVector()
        n_points = vec_src.n_points
        div = (pow(self.a, 2) + pow(self.b, 2))
        x11 = self.a / div
        x12 = self.b / div
        x21 = -self.b / div
        x22 = self.a / div
        dst = np.zeros([n_points, 2])

        for i in range(n_points):
            x, y = vec_src.get_point(i)
            x -= self.x_t
            y -= self.y_t
            dst[i, 0] = x11 * x + x12 * y
            dst[i, 1] = x21 * x + x22 * y
        vec_dst.set_from_points_array(dst)

        return vec_dst

    def set_transform_by_align(self, v, v_p):
        """
        :param v:
        :type v: ShapeVector
        :param v_p:
        :type v_p: ShapeVector
        :return: None
        """
        n_points = v.n_points
        self.a = np.dot(v_p.vector, v.vector) / np.dot(v.vector, v.vector)
        self.b = 0
        for i in range(n_points):
            x, y = v.get_point(i)
            x_p, y_p = v_p.get_point(i)
            self.b += (x * y_p - y * x_p)
        self.b = self.b / np.dot(v.vector, v.vector)
        self.x_t = -self.a * v.get_x_mean() + self.b * v.get_y_mean() + v_p.get_x_mean()
        self.y_t = -self.b * v.get_x_mean() - self.a * v.get_y_mean() + v_p.get_y_mean()

    def warp_image(self, img_src, img_dst):
        """
        :param img_src: source image
        :type img_src: numpy.ndarray
        :param img_dst: destination image
        :type img_dst: numpy.ndarray
        :return: warped image
        :rtype: numpy.ndarray
        """
        warp_mat = np.array([[self.a, -self.b, self.x_t], [self.b, self.a, self.y_t]])
        img_dst = cv.warpAffine(img_src, warp_mat, [img_src.shape[1], img_src.shape[0]])

        return img_dst

    def warp_image_back(self, img_src, img_dst, dst_size):
        """
        :param img_src: source image
        :type img_src: numpy.ndarray
        :param img_dst: destination image
        :type img_dst: numpy.ndarray
        :param dst_size: if destination image size should be used while warping
        :type dst_size: bool
        :return: warped image
        :rtype: numpy.ndarray
        """
        warp_mat = np.array([[self.a, -self.b, self.x_t], [self.b, self.a, self.y_t]])
        if dst_size:
            img_dst = cv.warpAffine(img_src, warp_mat, [img_dst.shape[1], img_dst.shape[0]])
        else:
            img_dst = cv.warpAffine(img_src, warp_mat, [img_src.shape[1], img_src.shape[0]])

        return img_dst


if __name__ == '__main__':

    a = np.array([[1, 6], [2, 7], [3, 8], [4, 9], [5, 10]])
    sv = ShapeVector()
    sv.set_from_points_array(a)
    print(f"points count: {sv.n_points}\n")
    print(f"vector of points:\n {sv.vector}")
    print(f"array of points:\n {sv.to_points_array()}\n")
    print(f"X coordinates mean: {sv.get_x_mean()}")
    print(f"Y coordinates mean: {sv.get_y_mean()}\n")
    sv.move_to_origin()
    print(f"vector moved to origin:\n {sv.vector}")
    sv.scale(2)
    print(f"vector scaled double:\n {sv.vector}")
