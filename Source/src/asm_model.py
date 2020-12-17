from shape_model import ShapeModel, FitResult, ModelImage
from shape_vector import ShapeVector, SimilarityTransformation
from features import FeatureExtractor

import numpy as np
import cv2 as cv
from dataclasses import dataclass
import math


class ASMModel (ShapeModel):

    # inverted covariance matrix pyramids for each landmark point
    cov_mat_pyr_inv: list

    # mean vector pyramids for each landmark point
    mean_vec_pyr: list

    # mean shapes of PCA model pyramid
    pca_shape_pyr: list

    # eigenvectors of the model pyramid
    eigenvectors_pyr: list

    # eigenvalues of the model pyramid
    eigenvalues_pyr: list

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
        self.pca_shape_pyr = list()
        self.eigenvectors_pyr = list()
        self.eigenvalues_pyr = list()
        self.points_on_normal = points_on_normal
        self.search_points_on_normal = search_points_on_normal
        self.feature_extractor = FeatureExtractor(self.pyramid_level,
                                                  self.points_on_normal,
                                                  self.search_points_on_normal)

    def build_model(self):
        """
        Builds Active Shape Model structure

        :return: None
        """
        super().build_model()
        self.build_asm_structure()

    def build_asm_structure(self):
        """
        Builds specific Active Shape model structure with gaussian pyramid

        :return: None
        """
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

        for i in range(self.pyramid_level):
            for j in range(self.n_landmarks):
                for k in range(self.n_images):
                    f_list = feature_extractor_list[k].get_feature(self.training_images[k].points, j, i)
                    if k == 0:
                        features_matrix = np.zeros([self.n_images, len(f_list)])
                    for ind, f in enumerate(f_list):
                        features_matrix[k, ind] = f

                cov_matrix = np.cov(features_matrix, bias=False)
                cov_matrix = np.linalg.svd(cov_matrix)
                mean = np.mean(features_matrix, axis=1)
                self.cov_mat_pyr_inv[i].append(cov_matrix)
                self.mean_vec_pyr[i].append(mean)

            self.pca_shape_pyr.append(self.pca_shape)
            self.eigenvectors_pyr.append(self.eigenvectors)
            self.eigenvalues_pyr.append(self.eigenvalues)

        feature_extractor_list.clear()

    def fit_all(self, img, top_left, size):
        """
        fits all points to given image

        :param img: target image
        :type img: numpy.ndarray
        :param top_left: top left corner of search area rectangle in the image
        :type top_left: (int, int)
        :param size: size of the search area rectangle
        :type size: (int, int)
        :return: result of fitting
        :rtype: ASMFitResult
        """
        x = top_left[0]
        y = top_left[1]
        w = size[1]
        h = size[0]
        x -= size[0]
        y -= size[1]
        if x < 0:
            x = 0
        if y < 0:
            y = 0
        if x + size[0] > img.shape[1]:
            w = img.shape[1] - x
        if y + size[1] > img.shape[0]:
            h = img.shape[0] - y

        fit_result = self.fit(img[y:y + h, x:x + w])
        s2 = SimilarityTransformation()
        s2.x_t = x
        s2.y_t = y
        s2.a = 1
        fit_result.similarity_trans = s2.multiply(fit_result.similarity_trans)

        return fit_result

    def fit(self, img):
        """
        fits the model to given image

        :param img: target image
        :type img: numpy.ndarray
        :return: result of fitting
        :rtype: ASMFitResult
        """
        # make sure image is in greyscale
        grey_img = img.copy()
        if len(grey_img.shape) == 3:
            grey_img = cv.cvtColor(grey_img, cv.COLOR_BGR2GRAY)

        # scale down the image
        ratio = math.sqrt(40000 / (grey_img.shape[0] * grey_img.shape[1]))
        resized_img = cv.resize(grey_img, (grey_img.shape[0] * ratio, grey_img.shape[1] * ratio))

        # create temporary search model image
        cur_search = ModelImage()
        cur_search.shape_info = self.shape_info
        cur_search.image = resized_img

        # create fit result object for search
        params = np.zeros([self.eigenvalues.shape[0], 1])
        sv = cur_search.shape_vector
        sv.vector = self.project_param_to_shape(params)
        st = sv.get_shape_transform_fitting_size(resized_img.shape)
        fit_result = ASMFitResult(params, st, self)
        cur_search.build_from_shape_vector(st)

        total_offset: int    # sum of offsets of current iteration
        shape_old = ShapeVector()
        self.feature_extractor.load_image(resized_img)

        for level in range(self.pyramid_level, -1, -1):
            for run in range(10):
                shape_old.set_from_points_array(cur_search.points)  # store old shape
                total_offset = 0
                best_ep = np.array([self.n_landmarks], np.int)

                for i in range(self.n_landmarks):
                    cur_best = -1
                    best_i = 0
                    candidate_points, features = \
                        self.feature_extractor.get_candidates_with_feature(cur_search.points, i, level)

                    for j in range(len(candidate_points)):
                        ct = cv.Mahalanobis(features[j], self.mean_vec_pyr[level][i], self.cov_mat_pyr_inv[level][i])

                        if ct < cur_best or cur_best < 0:
                            cur_best = ct
                            best_i = j
                            best_ep[i] = candidate_points[j]

                    total_offset += abs(best_i)

                for i in range(self.n_landmarks):
                    cur_search.points[i] = best_ep[i]
                    cur_search.points[i, 0] <<= level
                    cur_search.points[i, 1] <<= level
                    if level > 0:
                        cur_search.points[i, 0] += (1 << (level - 1))
                        cur_search.points[i, 1] += (1 << (level - 1))

                cur_search.shape_vector.set_from_points_array(cur_search.points)

                # TODO: implement finding parameters for the model

                cur_search.shape_vector = \
                    cv.PCABackProject(fit_result.params, self.pca_shape_pyr[level], self.eigenvectors_pyr[level])
                cur_search.build_from_shape_vector(fit_result.similarity_trans)

                avg_mov = total_offset / self.n_landmarks
                if avg_mov < 1.3:
                    run += 1
                    break

        st_ = SimilarityTransformation()
        st_.a = 1 / ratio
        fit_result.similarity_trans = st_.multiply(fit_result.similarity_trans)

        return fit_result


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
