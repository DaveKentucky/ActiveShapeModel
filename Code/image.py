import numpy as np
import cv2 as cv
import sys


class Image:

    def __init__(self, path, size=512):
        # path - relative track to image file
        # size - desired size of image (longer edge)

        # read image from file
        image = cv.imread(path)
        if image is None:
            sys.exit("Failed to load the image")

        self.__image = image  # jpeg image
        self.size = size    # desired max size of image
        self.points = []    # array for landmark points

        # set original size
        self.height = self.__image.shape[0]
        self.width = self.__image.shape[1]
        # print("w: " + str(self.width) + ", h: " + str(self.height))

        # scale the image ot desired size
        self.scale(self.width, self.height)

    def scale(self, w, h):
        # w - original image width in pixels
        # h - original image height in pixels

        if max(self.__image.shape[:2]) > self.size:
            print("scaling image...")
            if w == h:
                new_w = new_h = self.size
            elif w > h:
                scale = float(self.size / w)
                new_w = self.size
                new_h = int(h * scale)
            else:
                scale = float(self.size / h)
                new_w = int(w * scale)
                new_h = self.size

            self.__image = cv.resize(self.__image, (new_w, new_h))
            self.height = self.__image.shape[0]
            self.width = self.__image.shape[1]

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

        display_image = self.__image.copy()

        for point in self.points:
            x = point[0]
            y = point[1]
            index = self.points.index(point) + 1
            cv.circle(display_image, (x, y), 3, (200, 0, 0), -1)
            cv.putText(display_image, str(index), (x + 3, y - 3), cv.FONT_HERSHEY_PLAIN, 1.2, (200, 0, 0), 2)

        return display_image

    def get_landmarks_mask(self):
        # get the importance mask based on marked landmark points

        mask = np.zeros((self.height, self.width), np.uint8)

        for point in self.points:
            cv.circle(mask, point, 10, (255, 255, 255), -1)

        return mask

    def detect_SIFT(self):
        # detect and mark SIFT algorithm keypoints

        grey = cv.cvtColor(self.__image, cv.COLOR_BGR2GRAY)
        sift = cv.SIFT_create()
        kp = sift.detect(grey, self.get_landmarks_mask())

        img = cv.drawKeypoints(grey, kp, None)
        cv.imwrite("keypoints_sift.jpg", img)
        print("Saved result of SIFT algorithm to file")

    def detect_FAST(self):
        # detect and mark FAST algorithm keypoints

        grey = cv.cvtColor(self.__image, cv.COLOR_BGR2GRAY)
        fast = cv.FastFeatureDetector_create(threshold=25)
        kp = fast.detect(grey, self.get_landmarks_mask())

        img = cv.drawKeypoints(grey, kp, None, (200, 0, 0))
        cv.imwrite("keypoints_fast.jpg", img)
        print("Saved result of FAST algorithm to file")

    def detect_ORB(self):
        # detect and mark ORB algorithm keypoints

        grey = cv.cvtColor(self.__image, cv.COLOR_BGR2GRAY)
        orb = cv.ORB_create()
        kp = orb.detect(grey, self.get_landmarks_mask())

        img = cv.drawKeypoints(grey, kp, None, (200, 0, 0))
        cv.imwrite("keypoints_orb.jpg", img)
        print("Saved result of ORB algorithm to file")
