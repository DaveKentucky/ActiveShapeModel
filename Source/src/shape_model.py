from shape_info import ShapeInfo
from shape_vector import ShapeVector


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
    file_track: str
    # set of training images
    training_images: list

    # def align_all_shapes(self):

