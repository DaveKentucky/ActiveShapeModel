from shape_info import ShapeInfo, PointInfo
from shape_vector import ShapeVector, SimilarityTransformation

import numpy as np
import cv2 as cv
import sys


class ModelImage:

    # if image was loaded yet
    is_loaded: bool

    # number of landmark points in shape
    n_points: int

    # information about shape
    shape_info: ShapeInfo

    # array of landmark points
    points: np.ndarray

    # shape vectors
    shape_vector: ShapeVector

    # image
    image: np.ndarray

    # image name
    name: str

    def __init__(self):
        self.is_loaded = False
        self.n_points = 0
        self.shape_info = None
        self.points = None
        self.shape_vector = ShapeVector()

    def __repr__(self):
        if self.is_loaded:
            img_info = f"image of shape {self.image.shape} loaded from {self.name}"
        else:
            img_info = ""
        if self.n_points > 0:
            pts_info = f"{self.n_points} points marked:\n{self.points}"
        else:
            pts_info = ""
        return f"ModelImage: {img_info}, {pts_info}, image's shape vector:\n{self.shape_vector}"

    def read_from_file(self, directory, file):
        """
        Read training image from given file

        :param directory: path to directory with training images
        :type directory: str
        :param file: path to an image file
        :type file: str
        :return: None
        """
        if not self.is_loaded:
            path = directory + "/" + file
            image = cv.imread(path)
            if image is None:
                sys.exit("ModelImage: Failed to load the image")

            self.image = image
            self.name = file
            self.is_loaded = True

    def set_points_from_array(self, p):
        """
        Sets landmark points array with given array of points

        :param p: numpy array of points (Nx2 shape)
        :type p: numpy.ndarray
        :return: None
        """
        self.n_points = p.shape[0]
        self.points = p.copy()
        self.shape_vector.set_from_points_array(p)

    def set_points_from_list(self, lst):
        """
        Sets landmark points array with given list of points

        :param lst: list of numpy arrays of points
        :type lst: list
        :return: None
        """
        self.n_points = len(lst)
        self.points = np.zeros([self.n_points, 2], int)
        for i, point in enumerate(lst):
            self.points[i, 0] = point[0]
            self.points[i, 1] = point[1]
        self.shape_vector.set_from_points_array(self.points)

    def set_from_asf(self, directory, read_info):
        """
        Sets landmark points and shape info from an ASF file

        :param directory: path to directory with training images
        :type directory: str
        :param read_info: if the shape info should be created
        :type read_info: bool
        :return: shape info if created
        :rtype: ShapeInfo
        """
        filename = self.name.split('.')[0]

        file = open(directory + "/" + filename + ".asf", "r")
        lines = file.readlines()

        # get indices of lines with number of points in shape, first point data and name of the image file
        asf_indices = {'n_points': 0, 'points_start': 0, 'name': 0}
        for i, line in enumerate(lines):
            if line.find("number of model points") > -1:
                asf_indices['n_points'] = i + 2
            if line.find("model points") > -1:
                asf_indices['points_start'] = i + 4
            if line.find("host image") > -1:
                asf_indices['name'] = i + 2

        self.n_points = int(lines[asf_indices['n_points']])
        self.points = np.zeros([self.n_points, 2], int)

        # prepare containers for shape info data if it should be read
        if read_info:
            si = ShapeInfo()
            point_info_list = list()    # list of PointInfo objects for shape info
            start_id_list = list()   # list of starting indices of every contour for shape info
            type_list = list()     # list of type of every contour for shape info

        # read every point's coordinates
        index = asf_indices['points_start']     # index of the line with first point data
        for i in range(self.n_points):
            line = lines[index].split(sep="\t")
            coord_x = float(line[2])
            self.points[i, 0] = int(coord_x * self.image.shape[1])
            coord_y = float(line[3])
            self.points[i, 1] = int(coord_y * self.image.shape[0])
            index += 1

            # read other point's data if shape info should be read
            if read_info:
                id = int(line[0])
                type = int(line[1])
                c_from = int(line[5])
                c_to = int(line[6])
                pi = PointInfo(id, type, c_from, c_to)
                point_info_list.append(pi)

        self.shape_vector.set_from_points_array(self.points)

        # set shape info with read data
        if read_info:
            start_id = -1
            contours_count = 0
            for i, pi in enumerate(point_info_list):
                if pi.contour > start_id:
                    contours_count += 1
                    start_id = pi.contour
                    start_id_list.append(i)
                    if pi.type == 0:
                        type_list.append(1)
                    elif pi.type == 4:
                        type_list.append(0)
            si.n_contours = contours_count
            si.contour_start_index = start_id_list
            si.contour_is_closed = type_list
            si.point_info = point_info_list
            return si
        else:
            return None

    def show(self, show, win_name="image"):
        """
        Returns copy of the image with marked model shapes on it

        :param show: if the image should be displayed in a window
        :type show: bool
        :param win_name: name of the window for displaying an image
        :type win_name: str
        :return: image copy with model shapes
        :rtype: numpy.ndarray
        """
        if len(self.image.shape) != 3:
            img = cv.cvtColor(self.image, cv.COLOR_GRAY2RGB)
        else:
            img = self.image.copy()

        if self.shape_info is not None:
            img = self.shape_info.draw_points_on_image(img, self.points, draw_directly=False, labels=False)

        if show:
            cv.imshow(win_name, img)
            print("Press any key to continue...")
            cv.waitKey()
            cv.destroyWindow(win_name)

        return img

    def build_from_shape_vector(self, sim_trans):
        """
        Builds point list out of ShapeVector object

        :param sim_trans: similarity transformation of the shape
        :type sim_trans: SimilarityTransformation
        :return: None
        """
        self.n_points = self.shape_vector.n_points
        self.points = self.shape_vector.restore_to_point_list(sim_trans)

    def get_shape_frame(self, offset):
        """
        Get default bounding frame of the shape

        :param offset: offset of the frame
        :type offset: int
        :return: coordinates of top left point of the frame and its shape as tuple: (width, height)
        :rtype: ((int, int), (int, int))
        """
        top_left, size = self.shape_vector.get_bound_rectangle()
        x = int(top_left[0] - offset)
        if x < 0:
            x = 0
        y = int(top_left[1] - offset)
        if y < 0:
            y = 0
        w = int(size[0] + 2 * offset)
        if w > self.image.shape[1]:
            w = self.image.shape[1]
        h = int(size[1] + 2 * offset)
        if h > self.image.shape[0]:
            h = self.image.shape[0]
        return (x, y), (w, h)


if __name__ == '__main__':

    mi = ModelImage()
    mi.read_from_file('E:/Szkolne/Praca_inzynierska/ActiveShapeModel/Source/data/Face_images', 'face1.jpg')
    mi.show(True)
