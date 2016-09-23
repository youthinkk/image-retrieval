import math
import numpy as np


def euclidean_distance(feature1, feature2):
    feature1 = np.array(feature1)
    feature2 = np.array(feature2)
    sum_distance = np.sum(np.power(feature1 - feature2, 2.0))

    return math.sqrt(sum_distance)


def bhattacharyya_distance(feature1, feature2):
    """
    Calculate distance between two histograms using
    Bhattacharyya distance
    """
    feature1 = np.array(feature1)
    feature2 = np.array(feature2)
    sum1 = np.sum(feature1)
    sum2 = np.sum(feature2)

    root_sum = np.sum(np.sqrt(feature1 * feature2))
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


def color_similarity(feature1, feature2):
    return 1 - bhattacharyya_distance(feature1, feature2)


def sift_similarity(feature1, feature2):
    return 1 - chi2_distance(feature1, feature2)


def remove_duplicate(results):
    new_results = []
    image_id = {}
    regex = "_"

    for i in xrange(len(results)):
        score = results[i][0]
        file_name = results[i][1]
        unique_id = regex.join(file_name.split(regex)[:2])
        if unique_id not in image_id:
            image_id[unique_id] = None
            new_results.append((score, file_name))

    return new_results
