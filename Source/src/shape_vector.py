import procrustes_analysis as pa

import numpy as np


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

        :param v: vector of points
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
            for i, c in enumerate(self.vector):
                c += vec.vector[i]


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
