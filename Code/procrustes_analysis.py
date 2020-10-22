import numpy as np
import cv2 as cv
from math import sin, cos, atan
from scipy.linalg import norm


def get_translation(shape):
    """
    Calculates a translation for x and y axis that centers shape around the origin

    :param shape: an array containing x coordinates of shape points as first column
    and y coords as second column
    :type shape: numpy.ndarray
    :return: translation([x,y]) a NumPy array with x and y translation coordinates
    :rtype: numpy.array[int]
    """

    mean_x = np.mean(shape[::2]).astype(np.int)
    mean_y = np.mean(shape[1::2]).astype(np.int)

    return np.array([mean_x, mean_y])


def translate(shape):
    """
    Translates shape to the origin

    :param shape:  an array containing x coordinates of shape points as first column
    and y coords as second column
    :type shape: numpy.ndarray
    :return: None
    """

    mean_x, mean_y = get_translation(shape)
    shape[::2] -= mean_x
    shape[1::2] -= mean_y


def get_rotation_scale(reference_shape, shape):
    """
    Calculates rotation and scale that would optimally align shape with reference shape

    :param reference_shape: a shape that serves as reference for scaling and alignment
    :type reference_shape: numpy.ndarray
    :param shape: a shape that is scaled and aligned
    :type shape: numpy.ndarray
    :return: scale: a scaling factor, theta: a rotation angle in radians
    :rtype: (float, float)
    """

    a = np.dot(shape, reference_shape) / norm(reference_shape) ** 2

    # separate x and y for the sake of convenience
    ref_x = reference_shape[::2]
    ref_y = reference_shape[1::2]

    x = shape[::2]
    y = shape[1::2]

    b = np.sum(x * ref_y - ref_x * y) / norm(reference_shape) ** 2

    new_scale = np.sqrt(a ** 2 + b ** 2)
    theta = atan(b / max(a, 10 ** -10))  # avoid dividing by 0

    return round(new_scale, 1), round(theta, 2)


def get_rotation_matrix(theta):
    """
    Returns the object's rotation matrix

    :param theta: rotation angle in radians
    :type theta: float
    :return: rotation matrix
    :rtype: numpy.ndarray
    """
    return np.array([[cos(theta), -sin(theta)], [sin(theta), cos(theta)]])


def rotate(shape, theta):
    """
    Rotates a shape by angle theta Assumes a shape is centered around origin

    :param shape: a shape to be rotated
    :type shape: numpy.ndarray
    :param theta: angle in radians
    :type theta: float
    :return: a rotated shape
    :rtype: numpy.ndarray[float]
    """

    matrix = get_rotation_matrix(theta)

    # reshape so that dot product is easily computed
    temp_shape = shape.reshape((-1, 2)).T

    # rotate
    rotated_shape = np.dot(matrix, temp_shape)

    return rotated_shape.T.reshape(-1)


def procrustes_analysis(reference_shape, shape):
    """
    Scales, and rotates a shape optimally to be aligned with a reference shape

    :param reference_shape: a shape that serves as reference alignment
    :type reference_shape: numpy.ndarray
    :param shape: (2nx1 NumPy array) a shape that is aligned
    :type shape: numpy.ndarray
    :return: aligned_shape: an aligned shape translated to the location of reference shape
    :rtype: numpy.ndarray[float]
    """

    # copy both shapes in case originals are needed later
    temp_ref = np.copy(reference_shape)
    temp_sh = np.copy(shape)

    translate(temp_ref)
    translate(temp_sh)

    # get scale and rotation
    scale, theta = get_rotation_scale(temp_ref, temp_sh)

    # scale, rotate both shapes
    temp_sh = temp_sh / scale
    aligned_shape = rotate(temp_sh, theta)

    return np.int32(aligned_shape)


def procrustes_distance(reference_shape, shape):
    """
    Counts the square root of the sum of squared differences in the positions of the landmarks in two shapes

    :param reference_shape: a shape that serves as reference alignment
    :type reference_shape: numpy.ndarray
    :param shape: a shape that is aligned
    :type shape: numpy.ndarray
    :return: the distance between shapes
    :rtype: float
    """
    ref_x = reference_shape[::2]
    ref_y = reference_shape[1::2]

    x = shape[::2]
    y = shape[1::2]

    dist = np.sum(np.sqrt((ref_x - x) ** 2 + (ref_y - y) ** 2))

    print(dist.dtype)
    return dist


if __name__ == '__main__':
    # example code showing procrustes analysis of simple shapes

    background = np.zeros([400, 400, 3], np.uint8)
    first_shape = np.array([107, 205, 190, 216, 183, 121, 102, 110])
    second_shape = np.array([98, 203, 180, 212, 198, 101, 116, 99])
    colors = [(200, 0, 0), (0, 0, 200), (0, 200, 0)]

    first_points = first_shape.reshape((-1, 1, 2))
    second_points = second_shape.reshape((-1, 1, 2))

    cv.polylines(background, [first_points], True, colors[0], 2)
    cv.polylines(background, [second_points], True, colors[1], 2)

    x, y = get_translation(first_shape)

    new_shape = procrustes_analysis(first_shape, second_shape)
    new_shape[::2] = new_shape[::2] + x
    new_shape[1::2] = new_shape[1::2] + y

    points = new_shape.reshape((-1, 1, 2))

    cv.polylines(background, np.int32([points]), True, colors[2], 2)

    cv.imshow("procrustes", background)
    cv.waitKey(8000)
