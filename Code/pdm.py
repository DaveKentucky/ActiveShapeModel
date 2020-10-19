import numpy as np


class PDM:
    # point distribution model class

    def __init__(self, shape):
        array = np.array(shape)
        array = array.reshape(2 * len(shape))
        self.reference_shape = array
        # self.norm = self.normalized(self.reference_shape)
