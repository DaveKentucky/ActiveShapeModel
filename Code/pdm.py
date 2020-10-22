import numpy as np
import procrustes_analysis as procrustes
import cv2 as cv


class PDM:
    # point distribution model class

    def __init__(self, shape):
        array = np.array(shape)
        array = array.reshape(2 * len(shape))
        self.mean_shape = array
        self.distance = 0
        self.canvas = 255 * np.ones([512, 512, 3], np.uint8)
        self.draw_shape(list(array), 1)

    def add_shape(self, shape):
        """
        Performs procrustes analysis of given shape with reference mean shape
        Calculates a new mean shape and mean distance between shapes

        :param shape: a shape to be aligned to the mean
        :type shape: numpy.ndarray
        :return: None
        """
        array = np.array(shape)
        array = array.reshape(2 * len(shape))

        if len(array) == len(self.mean_shape):
            shapes_list = []
            shapes_list.append(self.mean_shape)
            shapes_list.append(array)
            shapes = np.zeros(np.array(shapes_list).shape)
            shapes[0] = shapes_list[0]
            shapes[1] = shapes_list[1]

            x, y = procrustes.get_translation(shapes[0])

            new_shape = procrustes.procrustes_analysis(self.mean_shape, array)
            new_shape[::2] = new_shape[::2] + x
            new_shape[1::2] = new_shape[1::2] + y

            shapes[1] = new_shape
            new_mean = np.mean(shapes, 0)
            new_distance = procrustes.procrustes_distance(new_mean, self.mean_shape)

            if new_distance != self.distance:
                new_mean = procrustes.procrustes_analysis(self.mean_shape, new_mean)
                new_mean[::2] = new_mean[::2] + x
                new_mean[1::2] = new_mean[1::2] + y
                self.mean_shape = new_mean
                self.distance = new_distance

        self.draw_shape(list(array), 2)

    def get_mean_shape(self):
        """
        Returns the mean shape of the PDM scaled into array of points
        :return: model's mean shape
        :rtype: numpy.ndarray[float, float]
        """
        return self.mean_shape.reshape(-1, 2)

    def draw_shape(self, shape, c):
        points = np.array(shape)
        points = points.reshape(-1, 2)

        if c == 0:
            color = (0, 0, 0)
        elif c == 1:
            color = (255, 0, 0)
        elif c == 2:
            color = (0, 255, 0)
        elif c == 3:
            color = (0, 0, 255)

        for point in points:
            x = point[0]
            y = point[1]
            cv.rectangle(self.canvas, (x - 1, y - 1), (x + 1, y + 1), color, -1)

    def save_mean_shape(self, filename):
        shape = list(self.mean_shape)
        self.draw_shape(shape, 0)
        cv.imwrite(filename, self.canvas)
