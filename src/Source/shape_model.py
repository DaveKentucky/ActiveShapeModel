from shape_info import ShapeInfo
from shape_vector import ShapeVector, SimilarityTransformation
from model_image import ModelImage

import cv2 as cv
import numpy as np
from dataclasses import dataclass


# dataclass of model fitting result
@dataclass
class FitResult:
    params: np.ndarray    # parameters for the model
    similarity_trans: SimilarityTransformation  # transformation to recover shape after fitting


class ShapeModel:

    # info about shape
    shape_info: ShapeInfo

    # mean shape of the model
    mean_shape: ShapeVector

    # number of images in training set
    n_images: int

    # number of landmark points in shape
    n_landmarks: int

    # track to the directory with training data
    directory: str

    # name tag of the model
    name_tag: str

    # set of training images
    training_images: list

    # mean shape of PCA model
    pca_shape: np.ndarray

    # eigenvectors of the model
    eigenvectors: np.ndarray

    # eigenvalues of the model
    eigenvalues: np.ndarray

    # parameter for restoring the shape
    sigma2: float

    # full PCA shape with parameters for restoring shape
    pca_full_shape: dict

    # level of the image pyramid constant
    pyramid_level = 3

    def __init__(self):
        self.shape_info = None
        self.mean_shape = None

    def __repr__(self):
        return f"Model '{self.name_tag}', training images: {self.n_images}, landmark points: {self.n_landmarks}"

    def build_model(self):
        """
        Builds model structure

        :return: None
        """
        print("\nBuilding model...")
        self.align_all_shapes()
        self.build_PCA()

    def align_all_shapes(self):
        """
        Translates, scales and rotates all the training images to a common coordinate frame

        :return: None
        """
        print("Aligning all training images...")
        for image in self.training_images:
            image.shape_vector.move_to_origin()

        # set first shape aligned to origin as new mean shape
        self.training_images[0].shape_vector.scale_to_one()
        new_mean = ShapeVector()
        new_mean.set_from_vector(self.training_images[0].shape_vector.vector)
        origin = ShapeVector()
        origin.set_from_vector(new_mean.vector)

        while True:
            current_mean = ShapeVector()
            current_mean.set_from_vector(new_mean.vector)
            new_mean = ShapeVector()
            new_mean.set_from_vector(np.zeros(current_mean.vector.shape, np.float))

            for image in self.training_images:
                image.shape_vector.align_to(current_mean)
                new_mean.add_vector(image.shape_vector)

            new_mean.vector /= self.n_images
            new_mean.align_to(origin)
            new_mean.scale_to_one()

            if np.linalg.norm(np.subtract(current_mean.vector, new_mean.vector), np.inf) < 1e-10:
                break

        self.mean_shape = current_mean

    def build_PCA(self):
        """
        Builds the model structure with Principal Component Analysis

        :return: None
        """
        print("Performing PCA...")
        length = self.training_images[0].shape_vector.n_points * 2
        pca_data = np.empty((self.n_images, length), np.float)
        for i in range(self.n_images):
            for j in range(length):
                pca_data[i, j] = self.training_images[i].shape_vector.vector[j]

        self.pca_shape, self.eigenvectors, self.eigenvalues = cv.PCACompute2(pca_data, mean=None, maxComponents=10)

        eigenvalues_sum = np.sum(self.eigenvalues)
        s_cur = 0
        for eigenvalue in self.eigenvalues:
            s_cur += eigenvalue[0]
            if s_cur > eigenvalues_sum * 0.98:
                break
        vd = self.training_images[0].shape_vector.n_points * 2
        self.sigma2 = (eigenvalues_sum - s_cur) / (vd - 4)
        self.pca_full_shape = {'mean': self.pca_shape,
                               'eigenvalues': self.eigenvalues,
                               'eigenvectors': self.eigenvectors}

    def set_shape_info(self, info):
        """
        Sets model's shape info object and propagates it for all the images in the model

        :type info: ShapeInfo
        :return: None
        """
        if self.shape_info is None:     # set shape info if it was not set before
            self.shape_info = info
            self.n_landmarks = len(info.point_info)
            for image in self.training_images:
                image.shape_info = info

    def read_train_data(self, directory, model_name, files):
        """
        Reads training data from directory

        :param directory: path to the directory with training images
        :type directory: str
        :param model_name: name of the model
        :type model_name: str
        :param files: list of files with training images
        :type files: list
        :return: None
        """
        self.directory = directory
        self.name_tag = model_name
        self.n_images = 0
        self.training_images = list()

        # read all files from target directory
        for filename in files:
            img = ModelImage()
            img.read_from_file(directory, filename)
            if img.is_loaded:
                # add read image to the list
                self.training_images.append(img)

        self.n_images = len(self.training_images)

    def set_from_asf(self):
        read_shape = True
        for i, image in enumerate(self.training_images):
            si = image.set_from_asf(self.directory, read_shape)
            if read_shape:
                self.set_shape_info(si)
                read_shape = False

    def project_param_to_shape(self, params_vec):
        """
        projects PCA params back to restore the shape and set it to given ShapeVector

        :param params_vec: vector of ASM parameters
        :type params_vec: numpy.ndarray
        :return: vector of points (2Nx1 array)
        :rtype numpy.ndarray
        """
        return cv.PCABackProject(params_vec, self.pca_shape, self.eigenvectors)

    def project_shape_to_param(self, shape_vec):
        """
        projects shape vector to ASM parameters

        :type shape_vec: ShapeVector
        :return: vector of parameters
        :rtype: numpy.ndarray
        """
        return cv.PCAProject(shape_vec, self.pca_shape, self.eigenvectors)

    def show_mean_shape(self, blank, image=None):
        """
        Draws a mean shape on the image

        :param blank: if the shape should be drawn on a blank canvas
        :type blank: bool
        :param image: canvas image
        :type image: numpy.ndarray
        :return: None
        """
        if blank:
            img = np.ones(self.training_images[0].image.shape)
        else:
            if image is not None:
                img = image.copy()
            else:
                img = self.training_images[0].image.copy()

        st = self.mean_shape.get_shape_transform_fitting_size(img.shape)
        v = self.mean_shape.restore_to_point_list(st)
        img = self.shape_info.draw_points_on_image(img, v, draw_directly=True, labels=False)
        cv.imshow("mean shape", img)
        print("Press any key to continue...")
        cv.waitKey()


if __name__ == '__main__':

    sm = ShapeModel()
