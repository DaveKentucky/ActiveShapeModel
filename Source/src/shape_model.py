from shape_info import ShapeInfo
from shape_vector import ShapeVector
from model_image import ModelImage

import os
import cv2 as cv
import numpy as np


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

    def set_shape_info(self, info):

        if self.shape_info is None:     # set shape info if it was not set before
            self.shape_info = info
            for image in self.training_images:
                image.shape_info = info
                if image.points is not None:    # clear previously saved points if any
                    image.points = None

    def read_train_data(self, directory, model_name):

        self.directory = directory
        self.name_tag = model_name
        self.n_images = 0
        self.training_images = list()

        # read all files from target directory
        for filename in os.listdir(directory):
            img = ModelImage()
            img.read_from_file(directory, filename)
            if img.is_loaded:
                # add read image to the list
                self.training_images.append(img)

        self.n_images = len(self.training_images)

    def align_all_shapes(self):

        for image in self.training_images:
            image.shape_vector.move_to_origin()

        # set first shape aligned to origin as new mean shape
        self.training_images[0].shape_vector.scale_to_one()
        new_mean = ShapeVector()
        new_mean.set_from_vector(self.training_images[0].shape_vector.vector)
        origin = new_mean.vector.copy()     # numpy 1D array

        while True:
            current_mean = ShapeVector()
            current_mean.set_from_vector(new_mean.vector)
            new_mean = ShapeVector()
            new_mean.set_from_vector(np.zeros(current_mean.vector.shape, np.float))

            for image in self.training_images:
                image.shape_vector.align_to(new_mean)
                new_mean.add_vector(image.shape_vector)

            new_mean.vector /= self.n_images
            new_mean.align_to(origin)
            new_mean.scale_to_one()

            if np.linalg.norm(np.subtract(current_mean.vector, new_mean.vector), np.inf) < 1e-10:
                break

        self.mean_shape = current_mean

    def build_PCA(self):

        length = self.training_images[0].shape_vector.vector.points[0]
        pca_data = np.empty((length, self.n_images), np.float)
        for i in range(self.n_images):
            for j in range(length):
                pca_data[j, i] = self.training_images[i].shape_vector[j]

        self.pca_shape, self.eigenvectors = cv.PCACompute(pca_data, mean=None)

    def build_model(self):

        self.align_all_shapes()
        self.build_PCA()


if __name__ == '__main__':

    sm = ShapeModel()
    sm.read_train_data('E:/Szkolne/Praca_inzynierska/ActiveShapeModel/Source/data/Face_images', 'face')
