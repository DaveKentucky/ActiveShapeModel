import numpy as np
import cv2 as cv
import sys
import math


class Image:

    def __init__(self, filename, size=512):
        # filename - image file from 'Code' directory
        # size - desired size of image (longer edge)

        # read image from file
        path = "./Data/" + filename + ".jpg"
        image = cv.imread(path)
        if image is None:
            sys.exit("Failed to load the image")

        self.__image = image  # original image
        self.size = size    # desired max size of image
        self.points = []    # array for landmark points

        # set original size
        self.height = self.__image.shape[0]
        self.width = self.__image.shape[1]

        # scale the image ot desired size
        self.image = self.prepare_image(self.width, self.height)    # scaled and greyscale image
        print("w: " + str(self.width) + ", h: " + str(self.height))

    def prepare_image(self, w, h):
        # w - original image width in pixels
        # h - original image height in pixels

        # size = self.size
        # if max(self.__image.shape[:2]) > size:
        #     if w == h:
        #         new_w = new_h = size
        #     elif w > h:
        #         scale = float(size / w)
        #         new_w = size
        #         new_h = int(h * scale)
        #     else:
        #         scale = float(size / h)
        #         new_w = int(w * scale)
        #         new_h = size

        image = self.__image
        if image.shape == 3:
            grey_image = cv.cvtColor(image, cv.COLOR_BGR2GRAY)
        else:
            grey_image = image

        ratio = math.sqrt(40000 / (w * h))
        new_w = int(w * ratio)
        new_h = int(h * ratio)
        image = cv.resize(grey_image, (new_w, new_h))

        self.height = image.shape[0]
        self.width = image.shape[1]
        return image

    def add_landmark_point(self, x, y):
        # add new point to the list of landmark points
        # and draw it on the displayed image

        self.points.append((x, y))

        print("landmark points marked:")
        for point in self.points:
            print(point)

    def remove_landmark_point(self):
        # remove the last landmark point from the list

        if len(self.points) > 0:
            del self.points[-1]
            print("landmark point deleted")
            if len(self.points) > 0:
                print("landmark points marked:")
                for point in self.points:
                    print(point)
        else:
            print("no landmark points marked \n")

    def get_display_image(self):
        # get image with notified landmark points on it

        display_image = self.image.copy()

        for point in self.points:
            x = point[0]
            y = point[1]
            index = self.points.index(point) + 1
            cv.circle(display_image, (x, y), 1, (200, 0, 0), -1)

        return display_image

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
