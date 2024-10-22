import numpy as np
import cv2
import math

COLOR_BINS = (8, 12, 3)


def describe_color(image):
    """
    Convert the image to the HSV color space and initialize
    the features used to quantify the image
    """
    image = cv2.cvtColor(image, cv2.COLOR_BGR2HSV)
    features = []

    # grab the dimensions and compute the center of the image
    (h, w) = image.shape[:2]
    (c_x, c_y) = (int(w * 0.5), int(h * 0.5))

    # divide the image into four rectangles/segments (top-left,
    # top-right, bottom-right, bottom-left)
    segments = [(0, c_x, 0, c_y), (c_x, w, 0, c_y), (c_x, w, c_y, h),
                (0, c_x, c_y, h)]

    # construct an elliptical mask representing the center of the
    # image
    (axes_x, axes_y) = (int(w * 0.75) / 2, int(h * 0.75) / 2)
    ellip_mask = np.zeros(image.shape[:2], dtype="uint8")
    cv2.ellipse(ellip_mask, (c_x, c_y), (axes_x, axes_y), 0, 0, 360, 255, -1)

    # loop over the segments
    for (start_x, end_x, start_y, end_y) in segments:
        # construct a mask for each corner of the image, subtracting
        # the elliptical center from it
        corner_mask = np.zeros(image.shape[:2], dtype="uint8")
        cv2.rectangle(corner_mask, (start_x, start_y), (end_x, end_y), 255, -1)
        corner_mask = cv2.subtract(corner_mask, ellip_mask)

        # extract a color histogram from the image, then update the
        # feature vector
        hist = histogram(image, corner_mask)
        features.extend(hist)

    # extract a color histogram from the elliptical region and
    # update the feature vector
    hist = histogram(image, ellip_mask)
    features.extend(hist)

    # return the feature vector
    return features


def histogram(image, mask):
    """
    Extract a 3D color histogram from the masked region of the
    image, using the supplied number of bins per channel; then
    normalize the histogram
    """
    hist = cv2.calcHist([image], [0, 1, 2], mask, COLOR_BINS,
                        [0, 180, 0, 256, 0, 256])
    hist = cv2.normalize(hist).flatten()

    # return the histogram
    return hist
