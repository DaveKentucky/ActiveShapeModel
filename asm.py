from pdm import PDM
import numpy as np
import cv2 as cv
import select_area as sa


if __name__ == '__main__':

    import sys

    # Set recursion limit
    sys.setrecursionlimit(10 ** 9)

    window = sa.DragRectangle

    image = cv.imread('Code/Data/Face_images/face4.jpg')
    width = image.shape[1]
    height = image.shape[0]
    sa.init(window, image, "Image", width, height)

    cv.namedWindow(window.window_name)
    cv.setMouseCallback(window.window_name, sa.drag_rectangle, window)

    # keep looping until rectangle finalized
    while True:
        # display the image
        cv.imshow(window.window_name, window.image)
        key = cv.waitKey(1) & 0xFF

        # if return_flag is True, break from the loop
        if window.return_flag:
            break

    print("Dragged rectangle coordinates")
    print(str(window.outRect.x) + ',' + str(window.outRect.y) + ',' +
          str(window.outRect.w) + ',' + str(window.outRect.h))

    # close all open windows
    cv.destroyAllWindows()

