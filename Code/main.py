import cv2 as cv
import image
from image import Image
from pdm import PDM


# create image object
filename = "Face_images/face1.jpg"
img = Image(filename)

cv.namedWindow("Image", cv.WINDOW_KEEPRATIO)
cv.setMouseCallback("Image", image.mouse_input, img)
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
    elif key == ord('n'):
        pdm = PDM(img.points)
        img = Image("Face_images/face3.jpg")
        img.set_landmarks_array(pdm)
        cv.setMouseCallback("Image", image.mouse_input, img)
        cv.imshow("Image", img.get_display_image())
    elif key == ord('s'):
        pdm.add_shape(img.points)
