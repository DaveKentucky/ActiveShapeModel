import numpy as np
import cv2 as cv
import select_area as sa
import procrustes_analysis as pa
import sys

# Set recursion limit
sys.setrecursionlimit(10 ** 9)


def get_shapes_bound(model):
    """
    Creates a bounding rectangle of the model

    :param model: model points array
    :type model: numpy.ndarray
    :return: array with bounding rectangle corner points
    :rtype: numpy.ndarray
    """
    left = np.min(model[::2])
    right = np.max(model[::2])
    top = np.min(model[1::2])
    bottom = np.max(model[1::2])

    return np.array([left, top, left, bottom, right, bottom, right, top])


def set_model_position(selected_box, model_box, model):
    """
    Sets model's position to fit marked box

    :param selected_box: array with marked rectangle corner points
    :type selected_box: numpy.ndarray
    :param model_box: array with original model's rectangle corner points
    :type model_box: numpy.ndarray
    :param model: model points array
    :type model: numpy.ndarray
    :return: model transformed to fit the marked box
    :rtype: numpy.ndarray
    """

    # align model with the selected area using procrustes analysis
    x, y = pa.get_translation(selected_box)
    aligned_box = pa.procrustes_analysis(selected_box, model_box)
    scale, theta = pa.get_scale_and_rotation(selected_box, model_box)

    # create array for aligned model and initialize it with distances from top left point of original model's box
    aligned_model = np.zeros(model.shape)
    aligned_model[::2] = model[::2] - model_box[0]
    aligned_model[1::2] = model[1::2] - model_box[1]

    # scale the distances and put into frame aligned using procrustes analysis
    aligned_model = aligned_model / scale
    aligned_model = pa.rotate(aligned_model, theta)
    aligned_model[::2] = aligned_model[::2] + x/2
    aligned_model[1::2] = aligned_model[1::2] + y/2

    print(model)
    print(aligned_model)

    return np.int32(aligned_model)


def set_initial_location(image, model):
    """
    Sets the initial location of the model on the image to area marked by the user

    :param image: image where the model should be applied
    :type image: numpy.ndarray
    :param model: model points array
    :type model: numpy.ndarray
    :return: model transformed to fit the marked box
    :rtype: numpy.ndarray
    """

    # create an instance of rectangle mark operator class
    select_window = sa.DragRectangle

    # read image's width and height
    width = image.shape[1]
    height = image.shape[0]

    # initiate the operator object
    sa.init(select_window, image, "Image", width, height)
    cv.namedWindow(select_window.window_name)
    cv.setMouseCallback(select_window.window_name, sa.drag_rectangle, select_window)
    cv.imshow("Image", select_window.image)

    # loop until selected area is confirmed
    wait_time = 1000
    while cv.getWindowProperty("Image", cv.WND_PROP_VISIBLE) >= 1:
        cv.imshow("Image", select_window.image)
        key = cv.waitKey(wait_time)
        if key == 27:
            cv.destroyWindow("Image")
        # if return_flag is True, break from the loop
        elif select_window.return_flag:
            break

    # save coordinates of selected area
    left, top, right, bottom = sa.get_area_rectangle(select_window)
    start_area = np.array([left, top, left, bottom, right, bottom, right, top])
    print("Selected box: ", start_area)

    # get bound rectangle of the model
    model_area = get_shapes_bound(model)
    print("Model's box: ", model_area)

    aligned_model = set_model_position(start_area, model_area, model)

    # # show aligned model on the image
    # points = aligned_model.reshape(-1, 2)
    # for point in points:
    #     x = point[0]
    #     y = point[1]
    #     cv.rectangle(image, (x - 1, y - 1), (x + 1, y + 1), (255, 255, 255), -1)
    #
    # cv.imshow("Image", image)
    # cv.waitKey(20000)

    return aligned_model


def get_gradient_image(image, method=0):
    """
    Creates an array of high gradient points of the given image

    :param image: original image in greyscale
    :type image: numpy.ndarray
    :param method: 0 - Laplacian gradient, 1 - Sobel gradient
    :type method: int
    :return: high gradient points image
    :rtype: numpy.ndarray
    """

    # blur the image
    blurred = cv.GaussianBlur(image, (5, 5), cv.BORDER_DEFAULT)

    # count gradients with selected method
    if method == 0:
        gradient = cv.Laplacian(blurred, cv.CV_64F)
    elif method == 1:
        sobel_x = cv.Sobel(blurred, cv.CV_64F, 1, 0, ksize=5)
        sobel_y = cv.Sobel(blurred, cv.CV_64F, 0, 1, ksize=5)
        gradient = cv.bitwise_or(sobel_x, sobel_y)
    elif method == 2:
        gradient = cv.Canny(blurred, 20, 30)
    else:
        return None

    return gradient


def get_surrounding_points_mean(image, x, y, d):
    """
    Calculates the mean value of d*d pixels surrounding point(x, y)

    :param image: original image in greyscale
    :type image: numpy.ndarray
    :param x: X coordinate of the middle point
    :type x: int
    :param y: Y coordinate of the middle point
    :type y: int
    :param d: diagonal length of the array of surrounding points (odd number)
    :type d: int
    :return: mean value of d*d surrounding pixels
    :rtype: numpy.float
    """

    # check if diagonal length is an odd value
    if d % 2 != 1:
        return None

    # get image size
    height = image.shape[0]
    width = image.shape[1]

    # set indices of nearby points array
    start_x = int(x - ((d - 1) / 2))
    start_y = int(y - ((d - 1) / 2))
    end_x = start_x + d
    end_y = start_y + d

    # validate indices with image size
    if start_x < 0:
        start_x = 0
    if start_y < 0:
        start_y = 0
    if end_x > width:
        end_x = width
    if end_y > height:
        end_y = height

    # calculate mean of the nearby points
    values = image[start_x:end_x, start_y:end_y]
    mean = np.mean(values)

    return mean


def find_new_point_position(image, x, y, search_range):
    """
    Finds new location for a point based on pixels' color mean values

    :param image: original image in greyscale
    :type image: numpy.ndarray
    :param x: X coordinate of the point
    :type x: int
    :param y: Y coordinate of the point
    :type y: int
    :param search_range: diagonal length of the search array (odd number)
    :type search_range: int
    :return: coordinates of the new point position
    :rtype: (int, int)
    """

    # check if search range is an odd value
    if search_range % 2 != 1:
        return None

    means = np.zeros((search_range, search_range), np.float)

    if search_range % 2 == 1:
        cor = (search_range - 1) / 2

    for i in range(search_range):
        for j in range(search_range):
            mx = x - cor + i
            my = y - cor + j
            means[i, j] = get_surrounding_points_mean(image, mx, my, 5)

    # find indices of the greatest mean
    index = [np.amax(np.argmax(means, axis=0)), np.amax(np.argmax(means, axis=1))]

    new_x = int(x - cor + index[0])
    new_y = int(y - cor + index[1])
    return new_x, new_y


def fit_model(image, model, search_range, gradient_method=0):

    original_model = model
    # convert image to greyscale
    grey_image = cv.cvtColor(image, cv.COLOR_BGR2GRAY)
    # set initial location of the model
    new_shape = set_initial_location(image, original_model)
    # get gradient image for searching
    search_image = get_gradient_image(grey_image, gradient_method)

    points = np.array(new_shape)
    points = points.reshape(-1, 2)
    for point in points:
        x = point[0]
        y = point[1]
        cv.rectangle(image, (x - 1, y - 1), (x + 1, y + 1), (0, 255, 0), -1)

    for k in range(20):
        # iterate each point in the model
        for i, (x, y) in enumerate(zip(new_shape[0::2], new_shape[1::2])):
            new_x, new_y = find_new_point_position(search_image, x, y, search_range)
            new_shape[2 * i] = new_x
            new_shape[2 * i + 1] = new_y

    points = np.array(new_shape)
    points = points.reshape(-1, 2)
    for point in points:
        x = point[0]
        y = point[1]
        cv.rectangle(image, (x - 1, y - 1), (x + 1, y + 1), (0, 0, 255), -1)
    cv.imshow("image", image)
    cv.waitKey(10000)


if __name__ == '__main__':

    mdl = np.array([130, 150, 150, 210, 240, 350, 330, 210, 350, 150])
    img = cv.imread('Data/Face_images/face3.jpg')

    fit_model(img, mdl, 33, gradient_method=0)

    # close all open windows
    cv.destroyAllWindows()

