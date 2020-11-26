import numpy as np


class ShapeVector:

    # vector of points
    vector: np.array

    # number of points in vector
    n_points: int

    def __init__(self, p):
        """
        Initializes vector with array of points

        :param p: 2D numpy array of points
        :type p: numpy.ndarray
        """
        self.set_from_points_array(p)

    def set_from_points_array(self, p):
        """
        Sets vector with array of points

        :param p: 2D numpy array of points
        :type p: numpy.ndarray
        """
        self.n_points = len(p[0])
        array = np.zeros([2 * self.n_points], np.int)
        for i, point in enumerate(p.T):
            array[2 * i] = point[0]
            array[2 * i + 1] = point[1]

        self.vector = array

    def get_point(self, i):
        """
        Returns coordinates of point at given index

        :param i: index of the point
        :return: tuple of point coordinates (X, Y)
        :rtype: (int, int)
        """
        return self.vector[2 * i], self.vector[2 * i + 1]

    def get_x_mean(self):
        """
        :return: mean of the points' X coordinates
        :rtype: int
        """
        return int(np.mean(self.vector[0::2]))

    def get_y_mean(self):
        """
        :return: mean of the points' Y coordinates
        :rtype: int
        """
        return int(np.mean(self.vector[1::2]))

    def to_points_array(self):
        """
        Returns array of points

        :return: 2D array of points
        :rtype: numpy.ndarray
        """
        array = np.array([self.vector[0::2], self.vector[1::2]])

        return array

    def translate(self, vx, vy):
        """
        Translates the vector with given translation values
        :param vx: X translation
        :type vx: int
        :param vy: Y translation
        :type vy: int
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
        self.vector = np.multiply(self.vector, s).astype(int)

    def move_to_origin(self):
        """
        Translates the vector's center to origin
        :return: None
        """
        self.translate(-self.get_x_mean(), -self.get_y_mean())


if __name__ == '__main__':

    v = np.array([[1, 2, 3, 4, 5], [6, 7, 8, 9, 10]])
    sv = ShapeVector(v)
    print(f"points count: {sv.n_points}\n")
    print(f"vector of points:\n {sv.vector}")
    print(f"array of points:\n {sv.to_points_array()}\n")
    print(f"X coordinates mean: {sv.get_x_mean()}")
    print(f"Y coordinates mean: {sv.get_y_mean()}\n")
    sv.move_to_origin()
    print(f"vector moved to origin:\n {sv.vector}")
    sv.scale(2)
    print(f"vector scaled double:\n {sv.vector}")

