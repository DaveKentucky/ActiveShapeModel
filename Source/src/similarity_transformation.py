from shape_vector import ShapeVector

import numpy as np
import cv2 as cv
import math


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

    def inverted_transform(self, vec_src, vec_dst):
        """

        :param vec_src:
        :type vec_src: ShapeVector
        :param vec_dst:
        :type vec_dst: ShapeVector
        :return:
        :rtype: ShapeVector
        """
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
