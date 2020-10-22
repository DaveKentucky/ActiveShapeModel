import numpy as np
import procrustes_analysis as procrustes


class PDM:
    # point distribution model class

    def __init__(self, shape):
        array = np.array(shape)
        array = array.reshape(2 * len(shape))
        self.mean_shape = array
        self.distance = 0

    def add_shape(self, shape):
        """
        Performs procrustes analysis of given shape with reference mean shape
        Calculates a new mean shape and mean distance between shapes

        :param shape: a shape to be aligned to the mean
        :type shape: numpy.ndarray
        :return: None
        """

        if len(shape) == len(self.mean_shape):
            shapes = np.zeros((np.array(shape), 2))
            shapes[0] = self.mean_shape

            shapes[1] = procrustes.procrustes_analysis(self.mean_shape, shape)
            new_mean = np.mean(shapes, 0)
            new_distance = procrustes.procrustes_distance(new_mean, self.mean_shape)

            if new_distance != self.distance:
                new_mean = procrustes.procrustes_analysis(self.mean_shape, new_mean)

                self.mean_shape = new_mean
                self.distance = new_distance

    def get_mean_shape(self):
        """
        Returns the mean shape of the PDM scaled into array of points
        :return: model's mean shape
        :rtype: numpy.ndarray[float, float]
        """
        return self.mean_shape.reshape(-1, 2)
