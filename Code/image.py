import numpy as np
import cv2 as cv
import sys
import math


class Image:
    # an image object with its parameters and landmark points

    def __init__(self, filename):
        """
        Initializes the object with image read from file

        :param filename: name of the image file
        :type filename: str
        """

        # read image from file
        path = "./Data/" + filename
        image = cv.imread(path)
        if image is None:
            sys.exit("Failed to load the image")

        self.__image = image  # original image
        self.points = []  # array for landmark points
        self.image = np.ndarray

        # set original size
        self.height = self.__image.shape[0]
        self.width = self.__image.shape[1]

        # scale the image ot desired size
        self.prepare_image(self.width, self.height)  # scaled and greyscale image
        print("w: " + str(self.width) + ", h: " + str(self.height))

    def prepare_image(self, w, h):
        """
        Scales the image down to ease the model calculations and saves it to separate variable

        :param w: original image width in pixels
        :type w: int
        :param h: original image height in pixels
        :type h: int
        :return: None
        """
        # w - original image width in pixels
        # h - original image height in pixels

        image = self.__image
        if len(image.shape) == 3:
            grey_image = cv.cvtColor(image, cv.COLOR_BGR2GRAY)
        else:
            grey_image = image

        ratio = math.sqrt(160000 / (w * h))
        new_w = int(w * ratio)
        new_h = int(h * ratio)
        image = cv.resize(grey_image, (new_w, new_h))
        self.__image = cv.resize(self.__image, (new_w, new_h))

        self.height = image.shape[0]
        self.width = image.shape[1]
        self.image = image

    def add_landmark_point(self, x, y):
        """
        Adds a new landmark point to the array

        :param x: X coordinate of the landmark point
        :type x: int
        :param y: Y coordinate of the landmark point
        :type y: int
        :return: None
        """

        # check if there is no point with these coordinates added yet
        for point in self.points:
            if point == [x, y]:
                return

        self.points.append([x, y])

        print(self.points[len(self.points) - 1])

    def get_landmark_point(self, coords, i=-1):
        """
        Returns the landmark point at given index or close to given point of the image

        :param coords: point coordinates on the image
        :type coords: list[int]
        :param i: index of the landmark point in array
        :type i: int
        :return: landmark point coordinates
        :rtype: list[int]
        """
        if i >= 0:
            return self.points[i]

        x = coords[0]
        y = coords[1]
        for point in self.points:
            if x - 1 <= point[0] <= x + 1 and y - 1 <= point[1] <= y + 1:
                _list = point
                _list.append(self.points.index(point))
                return _list

    def set_landmark_point(self, coords, i):
        """
        Sets landmark point in array to given coordinates

        :param coords: new coordinates of the point
        :type coords: list[int]
        :param i: index of the landmark point in array
        :type i: int
        :return: None
        """
        if self.points[i] is not None:
            self.points[i] = coords

    def remove_landmark_point(self):
        """
        Removes the last landmark point from the array
        """

        if len(self.points) > 0:
            del self.points[-1]
            print("landmark point deleted")

    def set_landmarks_array(self, pdm):
        """
        Sets the landmark points array to given point distribution model's mean shape

        :param pdm: point distribution model object
        :type pdm: pdm.PDM
        :return: None
        """
        self.points = pdm.get_mean_shape().tolist()

    def get_display_image(self):
        """
        Returns copy of the image with marked landmark points on it

        :return: image copy with landmarks
        :rtype: numpy.ndarray
        """
        # get image with notified landmark points on it

        display_image = self.__image.copy()
        for point in self.points:
            index = self.points.index(point) + 1
            x = point[0]
            y = point[1]
            cv.rectangle(display_image, (x - 1, y - 1), (x + 1, y + 1), (200, 0, 0), -1)
            cv.putText(display_image, str(index), (x + 1, y - 1), cv.FONT_HERSHEY_PLAIN, 1.0, (200, 0, 0))

        return display_image

    # TODO remove not used functions
    def get_landmarks_mask(self):
        # get the importance mask based on marked landmark points

        mask = np.zeros((self.height, self.width), np.uint8)

        for point in self.points:
            cv.circle(mask, point, 10, (255, 255, 255), -1)

        return mask

    def convert_landmarks(self):
        # convert marked landmark points to keypoint objects

        grey = cv.cvtColor(self.__image, cv.COLOR_BGR2GRAY)
        kp = cv.KeyPoint_convert(self.points)
        img = cv.drawKeypoints(grey, kp, None, (200, 0, 0))
        cv.imwrite("keypoints_landmarks.jpg", img)

        self.describe_keypoints(kp)

    def describe_keypoints(self, kp):
        # create descriptors to given keypoints

        orb = cv.ORB_create()
        kp, ds = orb.compute(self.__image, kp)

        for d in ds:
            print(d)


g_coords = tuple
g_index = -1


def mouse_input(event, x, y, flags, img):
    """
    Resolve mouse input on the image

    :param event: OpenCV event
    :type event: cv2.EVENT
    :param x: X coordinate of the mouse cursor
    :type x: int
    :param y: Y coordinate of the mouse cursor
    :type y: int
    :param flags: additional flags
    :param img: image object
    :type img: Image
    :return: None
    """
    # resolve mouse input on image

    global g_coords, g_index

    if event == cv.EVENT_LBUTTONDOWN:
        g_coords = [x, y]
        point = img.get_landmark_point(g_coords)
        if point is not None:
            g_index = point[2]
            img.set_landmark_point(g_coords, g_index)

    if event == cv.EVENT_LBUTTONUP:
        if g_index == -1:
            img.add_landmark_point(x, y)
        else:
            img.set_landmark_point([x, y], g_index)
            g_index = -1

    if event == cv.EVENT_MOUSEMOVE:
        if g_index != -1:
            img.set_landmark_point([x, y], g_index)
            cv.imshow("Image", img.get_display_image())

    cv.imshow("Image", img.get_display_image())
