import math
import numpy as np

INTENSITY_RANGE = 256
DIMENSION = 64


def get_histogram(image):
    """
    Get histogram by converting RGB to YCrCb
    """
    height = image.shape[0]
    width = image.shape[1]
    image_size = height * width
    bins = np.zeros([DIMENSION * DIMENSION * DIMENSION])
    step = INTENSITY_RANGE / DIMENSION

    for i in xrange(height):
        for j in xrange(width):
            blue = image[i, j, 0]
            green = image[i, j, 1]
            red = image[i, j, 2]

            # Convert from RGB to YCrCb
            y = int(np.round(0 + 0.299 * red + 0.587 * green + 0.114 * blue))
            cb = int(np.round(128 - 0.16874 * red - 0.33126 * green + 0.50000 * blue))
            cr = int(np.round(128 + 0.50000 * red - 0.41869 * green - 0.08131 * blue))

            bin_y = y / step
            bin_cb = cb / step
            bin_cr = cr / step

            # Compute histogram of individual pixel
            bins[bin_y * DIMENSION * DIMENSION + bin_cb * DIMENSION + bin_cr] += 1

    # Normalise
    for i in xrange(bins.shape[0]):
        bins[i] = bins[i] / image_size

    return bins


def calculate_distance(histogram1, histogram2):
    """
    Calculate distance between two histograms using Bhattacharyya distance
    """
    length = histogram1.shape[0]
    sum1 = np.sum(histogram1)
    sum2 = np.sum(histogram2)
    root_sum = 0.

    for i in xrange(length):
        root_sum += math.sqrt(histogram1[i] * histogram2[i])

    square_distance = 1 - root_sum / math.sqrt(sum1 * sum2)
    square_distance = 0 if square_distance < 0 else square_distance

    return math.sqrt(square_distance)


def histogram_similarity(image1, image2):
    """
    Compare the similarity of two images
    """
    histogram1 = get_histogram(image1)
    histogram2 = get_histogram(image2)
    distance = calculate_distance(histogram1, histogram2)
    return 1 - distance