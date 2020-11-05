import numpy as np


def get_mean_vector(data):
    """
    Computes the mean of each row of the given array

    :param data: transposed array of samples
    :type data: numpy.ndarray
    :return: vector of the mean values
    :rtype: numpy.ndarray
    """

    n = data.shape[0]
    mean_vector = np.zeros([n, 1], np.float)

    for i in range(n):
        mean = np.mean(data[i])
        mean_vector[i] = mean

    return mean_vector


def get_eigenvalues_and_eigenvectors(data):
    """
    Computes the eigenvalues and eigenvectors of given data array

    :param data: transposed array of samples
    :type data: numpy.ndarray
    :return: arrays of eigenvalues and eigenvectors
    :rtype: (numpy.ndarray, numpy.ndarray)
    """

    cov_mat = np.cov([data[0, :], data[1, :], data[2, :], data[3, :], data[4, :], data[5, :]])
    print('Covariance Matrix:\n', cov_mat)

    eig_val, eig_vec = np.linalg.eig(cov_mat)

    for i in range(len(eig_val)):
        eigvec = eig_vec[:, i].reshape(1, len(data)).T

        print('Eigenvector {}: \n{}'.format(i + 1, eigvec))
        print('Eigenvalue {} from covariance matrix: {}'.format(i + 1, eig_val[i]))

    return eig_val, eig_vec


def get_desired_eigenvalues_count(eigenvalues, proportion):
    """
    Computes the amount of significant eigenvectors dependent to given proportion

    :param eigenvalues: array of eigenvalues
    :type eigenvalues: numpy.ndarray
    :param proportion: desired percentage of data stored in significant eigenvectors (between 0 and 1)
    :type proportion: float
    :return: number of significant eigenvectors
    :rtype: int
    """

    if 0 < proportion <= 1:

        # sort the array from largest to smallest
        eigenvalues = np.sort(eigenvalues)[::-1]

        # calculate sum of variances and the goal variance
        total_variance = np.sum(eigenvalues)
        goal_variance = proportion * total_variance

        variance = float(0)
        for i in range(len(eigenvalues)):
            new_variance = variance + eigenvalues[i]
            if goal_variance <= variance:
                return i + 1
            else:
                variance = new_variance
    else:
        return -1


def principal_component_analysis(samples, proportion):
    """
    Reduces the dimensionality of given array keeping at least given proportion of the data

    :param samples: array of samples that columns quantity should be reduced
    :type samples: numpy.ndarray
    :param proportion: desired percentage of data stored in returned eigenvectors (between 0 and 1)
    :type proportion: float
    :return: arrays of eigenvalues and eigenvectors
    :rtype: (numpy.ndarray, numpy.ndarray)
    """

    transposed = np.transpose(samples)
    mean = get_mean_vector(transposed)

    # Compute eigenvectors and eigenvalues
    eig_val, eig_vec = get_eigenvalues_and_eigenvectors(transposed)

    # Make a list of (eigenvalue, eigenvector) tuples
    eig_pairs = [(np.abs(eig_val[i]), eig_vec[:, i]) for i in range(len(eig_val))]

    # Sort the (eigenvalue, eigenvector) tuples from high to low
    eig_pairs.sort(key=lambda x: x[0], reverse=True)

    # Visually confirm that the list is correctly sorted by decreasing eigenvalues
    # for i in eig_pairs:
    #     print(i[0])

    t = get_desired_eigenvalues_count(eig_val, proportion)

    if t is not None:
        if t >= 0:
            desired_eigenvalues = np.zeros([t, 1], np.float)
            desired_eigenvectors = np.zeros([t, len(eig_vec)], np.ndarray)

            for i in range(t):
                desired_eigenvalues[i] = eig_pairs[i][0]
                desired_eigenvectors[i] = eig_pairs[i][1]

            return desired_eigenvalues, desired_eigenvectors
        else:
            return eig_val, eig_vec
    else:
        return eig_val, eig_vec

    # matrix_w = np.hstack((eig_pairs[0][1].reshape(3, 1), eig_pairs[1][1].reshape(3, 1)))
    # print('Matrix W:\n', matrix_w)
    #
    # transformed = matrix_w.T.dot(data)
    # print(transformed)


# example code showing procrustes analysis of simple shapes
if __name__ == '__main__':

    array = np.array([[1, 4, 2, 2, 8, 4], [2, 4, 3, 3, 7, 5], [1, 5, 2, 3, 9, 4], [2, 6, 2, 3, 8, 5]], np.int32)
    eigvals, eigvecs = principal_component_analysis(array, 0.99)

    print("\nEigenvalues:")
    print(eigvals)
    print("\nEigenvectors:")
    print(eigvecs)
