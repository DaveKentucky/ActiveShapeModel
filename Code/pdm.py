import numpy as np
import procrustes_analysis as procrustes
import cv2 as cv
import image
import database
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

        self.mean_shape = None  # the mean shape (1D array)
        self.shapes = None      # all shapes (array of 1D arrays)
        self.points_count = 0   # number of points in a single shape
        self.name = name        # name of the shape
        self.distance = None    # procrustes distance
        self.canvas = 255 * np.ones([512, 512, 3], np.uint8)

        if not self.read_from_db(self.name):
            self.build_from_images(directory, count, name)

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

        # add first shape as the mean shape
        if self.mean_shape is None:
            self.mean_shape = array
            self.points_count = len(shape)
            self.distance = 0
            self.draw_shape(list(array), color)
            self.shapes = array
        else:
            # check if the shapes' lengths are identical
            if len(shape) == self.points_count:
                # prepare data of a new shape and the mean shape for procrustes analysis
                shapes_list = []
                shapes_list.append(self.mean_shape)
                shapes_list.append(array)
                shapes = np.zeros(np.array(shapes_list).shape)
                shapes[0] = shapes_list[0]
                shapes[1] = shapes_list[1]

                # save mean shape's translation
                x, y = procrustes.get_translation(shapes[0])

                # run procrustes analysis on both shapes
                new_shape = procrustes.procrustes_analysis(self.mean_shape, array)

                # translate aligned shape to mean shape's location
                new_shape[::2] = new_shape[::2] + x
                new_shape[1::2] = new_shape[1::2] + y

                # save shapes
                self.shapes = np.vstack((self.shapes, new_shape))
                print(self.shapes.shape)
                shapes[1] = new_shape
                new_mean = np.mean(shapes, 0)
                new_distance = procrustes.procrustes_distance(new_mean, self.mean_shape)

                # check if procrustes distance has changed
                if new_distance != self.distance:
                    # update the mean shape including a new and an old mean
                    new_mean = procrustes.procrustes_analysis(self.mean_shape, new_mean)
                    new_mean[::2] = new_mean[::2] + x
                    new_mean[1::2] = new_mean[1::2] + y

                # save results
                self.mean_shape = new_mean
                self.distance = new_distance

                self.draw_shape(list(array), color)

                index = len(self.shapes)
                print(f"New image added to the model successfully\nImage index: {index}")

    def get_mean_shape(self):
        """
        Returns the mean shape of the PDM scaled into array of points

        :return: model's mean shape
        :rtype: numpy.ndarray[float, float]
        """

        return self.mean_shape.reshape(self.points_count, 2)

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

        # reshape shape into 2D array to enable drawing with OpenCV
        points = np.array(shape)
        points = points.reshape(self.points_count, 2)

        # pick a color
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

        # draw all points on canvas
        for point in points:
            x = point[0]
            y = point[1]
            cv.rectangle(canvas, (x - 1, y - 1), (x + 1, y + 1), color, -1)

    def save_to_jpg(self, filename):
        """
        Saves model into 2 jpg images (mean shape and all shapes)

        :param filename: name of the target file
        :type filename: str
        :return: None
        """

        # save an image of the mean shape
        shape = list(self.mean_shape)
        self.draw_shape(shape, 0)
        cv.imwrite("all_shapes.jpg", self.canvas)

        # save an image of all shapes
        self.canvas = 255 * np.ones([512, 512, 3], np.uint8)
        self.draw_shape(shape, 0)
        cv.imwrite(filename, self.canvas)

        print(f"Mean shape saved to file: {filename}\nAll shapes saved to file all_shapes.jpg")

    def save_to_db(self):
        """
        Saves model into database
        """

        my_db, my_cursor = database.create_database()
        result_message = database.insert_pdm(my_db, my_cursor, self.shapes, self.points_count, self.name)
        if result_message is not None:
            print(result_message)

    def build_from_images(self, directory, count, name):
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

        self.save_to_jpg("mean_shape.jpg")

    def read_from_db(self, name):
        """
        Read model from database

        :param name: name of the model
        :type name: str
        :return: if the model was read from database
        :rtype: bool
        """
        my_db, my_cursor = database.connect_to_database()
        db_read = database.get_pdm(my_cursor, name)
        if db_read is not None:
            self.mean_shape, self.shapes, self.points_count = db_read
            return True
        else:
            return False
