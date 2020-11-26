import numpy as np


class ShapeVector:

    # 2D array of points
    vector: np.array

    def __init__(self, p):

        self.vector = p.copy()

    def get_points_count(self):
        return self.vector.shape[1]

    def get_point(self, i):
        return self.vector[:, i]

    def get_x_mean(self):
        return int(np.mean(self.vector[0, :]))

    def get_y_mean(self):
        return int(np.mean(self.vector[1, :]))

    def translate(self, vx, vy):

        self.vector[0] += vx
        self.vector[1] += vy

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

