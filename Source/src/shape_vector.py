import numpy as np


class ShapeVector:

    # vector of points
    vector: np.array
    # number of points in vector
    n_points: int

    def __init__(self, p):

        self.n_points = len(p[0])
        array = np.zeros([2 * self.n_points], np.int)
        for i, point in enumerate(p.T):
            array[2 * i] = point[0]
            array[2 * i + 1] = point[1]

        self.vector = array

    def get_points_count(self):
        return len(self.vector) / 2

    def get_point(self, i):
        return self.vector[2 * i], self.vector[2 * i + 1]

    def get_x_mean(self):
        return int(np.mean(self.vector[0::2]))

    def get_y_mean(self):
        return int(np.mean(self.vector[1::2]))

    def translate(self, vx, vy):

        for i in range(self.n_points):
            self.vector[2 * i] += vx
        for i in range(self.n_points):
            self.vector[2 * i + 1] += vy

    def move_to_origin(self):
        self.translate(-self.get_x_mean(), -self.get_y_mean())

    def scale(self, s):
        self.vector = np.multiply(self.vector, s)


if __name__ == '__main__':

    v = np.array([[1, 2, 3, 4, 5], [6, 7, 8, 9, 10]])
    sv = ShapeVector(v)
    print(f"points count: {sv.get_points_count()}")
    print(f"X coordinates mean: {sv.get_x_mean()}")
    print(f"Y coordinates mean: {sv.get_y_mean()}")
    sv.move_to_origin()
    print(f"vector moved to origin:\n {sv.vector}")
    sv.scale(2)
    print(f"vector scaled double:\n {sv.vector}")

