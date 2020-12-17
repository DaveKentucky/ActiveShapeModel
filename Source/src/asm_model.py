from shape_model import ShapeModel, FitResult
from shape_vector import ShapeVector
from features import FeatureExtractor

import numpy as np
import cv2 as cv
from dataclasses import dataclass


class ASMModel (ShapeModel):

    # inverted covariance matrix pyramids for each landmark point
    cov_mat_pyr_inv: list

    # mean vector pyramids for each landmark point
    mean_vec_pyr: list

    # mean shapes of PCA model pyramid
    pca_shape_pyr: list

    # eigenvectors of the model pyramid
    eigenvectors_pyr: list

    # number of points to take in count on normal vector to the shape
    points_on_normal: int

    # number of points to take in count on normal vector to the shape while searching
    search_points_on_normal: int

    # feature extractor object for searching points in the image
    feature_extractor: FeatureExtractor

    def __init__(self, points_on_normal=4, search_points_on_normal=6):
        super().__init__()
        self.cov_mat_pyr_inv = list()
        self.mean_vec_pyr = list()
        self.points_on_normal = points_on_normal
        self.search_points_on_normal = search_points_on_normal
        self.feature_extractor = FeatureExtractor(self.pyramid_level,
                                                  self.points_on_normal,
                                                  self.search_points_on_normal)

    def build_model(self):

        super().build_model()
        self.build_asm_structure()

    def build_asm_structure(self):

        self.feature_extractor.shape_info = self.shape_info

        feature_extractor_list = list()
        for image in self.training_images:
            fe = FeatureExtractor(self.pyramid_level,
                                  self.points_on_normal,
                                  self.search_points_on_normal,
                                  self.shape_info)
            fe.load_image(image.image)
            feature_extractor_list.append(fe)

        for i in range(self.pyramid_level):
            self.cov_mat_pyr_inv.append(list())
            self.mean_vec_pyr.append(list())

        for i in range(self.pyramid_level + 1):
            for j in range(self.n_landmarks):
                for k in range(self.n_images):
                    f_list = feature_extractor_list[k].get_feature(self.training_images[k].points, j, i)
                    if k == 0:
                        features_matrix = np.zeros([self.n_images, len(f_list)])
                    for ind, f in enumerate(f_list):
                        features_matrix[k, ind] = f
                cov_matrix = np.cov(features_matrix, bias=False)
                cov_matrix = np.linalg.inv(cov_matrix)
                mean = np.mean(features_matrix, axis=1)
                self.cov_mat_pyr_inv[i].append(cov_matrix)
                self.mean_vec_pyr[i].append(mean)

        feature_extractor_list.clear()


@dataclass
class ASMFitResult (FitResult):
    asm_model: ASMModel     # Active Shape Model

    def to_point_list(self, pts_vec):
        sv = ShapeVector()
        sv.vector = self.asm_model.project_param_to_shape(self.params)
        sv.restore_to_point_list(pts_vec, self.similarity_trans)


if __name__ == '__main__':

    # asm = ASMModel()
    arr1 = [1, 2, 3]
    arr2 = [4, 5, 6]
    arr3 = [7, 8, 9]
    arr4 = [10, 11, 12]
    mat = np.array([arr1, arr2, arr3, arr4])
    print(mat)
    print(np.cov(mat, bias=False))
