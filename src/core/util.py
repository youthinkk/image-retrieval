import math
import numpy as np


def bhattacharyya_distance(feature1, feature2):
    """
    Calculate distance between two histograms using
    Bhattacharyya distance
    """
    length = len(feature1)
    sum1 = np.sum(feature1)
    sum2 = np.sum(feature2)
    root_sum = 0.

    for i in xrange(length):
        root_sum += math.sqrt(feature1[i] * feature2[i])

    square_distance = 1 - root_sum / math.sqrt(sum1 * sum2)
    square_distance = 0 if square_distance < 0 else square_distance

    return math.sqrt(square_distance)


def chi2_distance(feature1, feature2, eps=1e-10):
    # compute the chi-squared distance
    feature1 = np.array(feature1)
    feature2 = np.array(feature2)

    numerator = np.square(feature1 - feature2).astype(float)
    denominator = np.add(feature1 + feature2, eps).astype(float)
    distance = 0.5 * np.sum(numerator/denominator)

    # return the chi-squared distance
    return distance


def compute_similarity(feature1, feature2):
    """
    Compare the similarity of two images
    """
    return 1 - chi2_distance(feature1, feature2)
