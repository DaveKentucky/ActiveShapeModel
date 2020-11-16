import numpy as np
import procrustes_analysis as procrustes
import cv2 as cv
import image
from image import Image, mouse_input


class PDM:
    # point distribution model class

    def __init__(self, directory, count, name):
        """
        Initializes the point distribution model object with set of given positive images.
        The files should have .jpg extension and share identical name with incrementing number suffix
        ( ex. image1, image2, image3, ... )

        :param directory: name of the directory including images
        :type directory: str
        :param count: number of images to load
        :type count: int
        :param name: common name of all the images to load
        :type name: str
        """
        self.mean_shape = None
        self.shapes = None
        self.distance = None
        self.canvas = 255 * np.ones([512, 512, 3], np.uint8)

        self.build_model_from_images(directory, count, name)

    def add_shape(self, shape, color):
        """
        Performs procrustes analysis of given shape with reference mean shape
        Calculates a new mean shape and mean distance between shapes

        :param shape: a shape to be aligned to the mean
        :type shape: numpy.ndarray
        :return: None
        """
        array = np.array(shape)
        array = array.reshape(2 * len(shape))

        if self.mean_shape is None:
            self.mean_shape = array
            self.distance = 0
            self.draw_shape(list(array), color)
            self.shapes = array
        else:
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

                self.shapes = np.append([self.shapes], [new_shape], axis=0)
                print(self.shapes.shape)
                shapes[1] = new_shape
                new_mean = np.mean(shapes, 0)
                new_distance = procrustes.procrustes_distance(new_mean, self.mean_shape)

                if new_distance != self.distance:
                    new_mean = procrustes.procrustes_analysis(self.mean_shape, new_mean)
                    new_mean[::2] = new_mean[::2] + x
                    new_mean[1::2] = new_mean[1::2] + y

                self.mean_shape = new_mean
                self.distance = new_distance

            self.draw_shape(list(array), color)

    def get_mean_shape(self):
        """
        Returns the mean shape of the PDM scaled into array of points

        :return: model's mean shape
        :rtype: numpy.ndarray[float, float]
        """
        return self.mean_shape.reshape(-1, 2)

    def draw_shape(self, shape, c, canvas=None):
        """
        Draws the points from given shape on the object's canvas

        :param shape: shape to be drawn (2D array)
        :type shape: list
        :param c: color code
        :type c: int
        :param canvas: target image to draw shape on (object's canvas image if None)
        :type canvas: numpy.ndarray
        :return: None
        """
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

        if canvas is None:
            canvas = self.canvas

        for point in points:
            x = point[0]
            y = point[1]
            cv.rectangle(canvas, (x - 1, y - 1), (x + 1, y + 1), color, -1)

    def save_mean_shape(self, filename):
        """
        Saves the mean shape of the model to file.
        Saves an image of all shapes added to additional file

        :param filename: name of the target file
        :type filename: str
        :return: None
        """
        shape = list(self.mean_shape)
        self.draw_shape(shape, 0)
        cv.imwrite("all_shapes.jpg", self.canvas)

        self.canvas = 255 * np.ones([512, 512, 3], np.uint8)
        self.draw_shape(shape, 0)
        cv.imwrite(filename, self.canvas)

    def build_model_from_images(self, directory, count, name):
        """
        Loads given set of images and runs manual labeling of points distribution model based on the images

        :param directory: name of the directory including images
        :type directory: str
        :param count: number of images to load
        :type count: int
        :param name: common name of all the images to load
        :type name: str
        :return: None
        """
        cv.namedWindow("Image", cv.WINDOW_KEEPRATIO)
        cv.namedWindow("Help image", cv.WINDOW_KEEPRATIO)

        for i in range(count):
            index = i + 1
            if i > 0:
                help_img = img
                cv.imshow("Help image", help_img.get_display_image())

            filename = str(directory + "/" + name + str(index) + ".jpg")
            img = Image(filename)

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
                    color = (i % 3) + 1
                    self.add_shape(img.points, color)
                    break

        self.save_mean_shape("mean_shape.jpg")
