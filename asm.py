import numpy as np
import cv2 as cv
import select_area as sa
import procrustes_analysis as pa
import sys

# Set recursion limit
sys.setrecursionlimit(10 ** 9)


def get_shapes_bound(model):

    left = np.min(model[::2])
    right = np.max(model[::2])
    top = np.min(model[1::2])
    bottom = np.max(model[1::2])

    return np.array([left, top, left, bottom, right, bottom, right, top])


def set_new_bounds(selected_box, model_box, model):

    # align model with the selected area using procrustes analysis
    x, y = pa.get_translation(selected_box)
    aligned_box = pa.procrustes_analysis(selected_box, model_box)
    scale, theta = pa.get_scale_and_rotation(selected_box, model_box)
    aligned_box[::2] = aligned_box[::2] + x
    aligned_box[1::2] = aligned_box[1::2] + y

    # create array for aligned model and initialize it with distances from top left point of original model's box
    aligned_model = np.zeros(model.shape)
    aligned_model[::2] = model[::2] - model_box[0]
    aligned_model[1::2] = model[1::2] - model_box[1]

    # scale the distances and put into frame aligned using procrustes analysis
    aligned_model = aligned_model / scale
    aligned_model[::2] = aligned_model[::2] + aligned_box[0]
    aligned_model[1::2] = aligned_model[1::2] + aligned_box[1]

    print(model)
    print(aligned_model)

    return np.int32(aligned_model)


def set_initial_location(image, model):

    select_window = sa.DragRectangle

    width = image.shape[1]
    height = image.shape[0]

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

    aligned_model = set_new_bounds(start_area, model_area, model)

    cv.imshow("Image", image)
    cv.waitKey(20000)


if __name__ == '__main__':

    model = np.array([130, 150, 150, 210, 240, 350, 330, 210, 350, 150])
    image = cv.imread('Code/Data/Face_images/face4.jpg')
    # if image is not None:

    set_initial_location(image, model)

    # close all open windows
    cv.destroyAllWindows()

