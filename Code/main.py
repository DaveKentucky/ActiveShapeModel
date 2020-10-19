import cv2 as cv
from image import Image
from pdm import PDM


def mouse_click(event, x, y, flags, param):
    # resolve mouse click on image

    # left button clicked
    if event == cv.EVENT_LBUTTONDOWN:
        point = (x, y)
        # print(point)

        # add landmark point to image's list
        img.add_landmark_point(x, y)
        cv.imshow("Image", img.get_display_image())

# create image object
filename = "Face_images/face1.jpg"
img = Image(filename)

cv.namedWindow("Image", cv.WINDOW_KEEPRATIO)
cv.setMouseCallback("Image", mouse_click, filename)
cv.imshow("Image", img.get_display_image())

wait_time = 1000
while cv.getWindowProperty("Image", cv.WND_PROP_VISIBLE) >= 1:
    key = cv.waitKey(wait_time)
    if key == 27:
        cv.destroyWindow("Image")
    elif key == ord('r'):
        img.remove_landmark_point()
        cv.imshow("Image", img.get_display_image())
    elif key == ord('k'):
        img.convert_landmarks()
    elif key == ord('p'):
        pdm = PDM(img.points)
        print(pdm.reference_shape)
