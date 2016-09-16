import cv2
import glob
import csv
import numpy as np
from histogram import ColorDescriptor, calculate_distance
from learning import inference
from keywords import runSIFT

COLOR_BINS = (8, 12, 3)

color_descriptor = ColorDescriptor(COLOR_BINS)

if __name__ == '__main__':
    IMAGE1_PATH = "../data/test/0001_127194972_balloons.jpg"
    IMAGE2_PATH = "../data/test/0001_439648413_alley.jpg"
    image1 = cv2.imread(IMAGE1_PATH)
    image2 = cv2.imread(IMAGE2_PATH)

    # Compute color similarity and return a scalar similarity score
    color_similarity = color_descriptor.similarity(image1, image1)

    # Compute prediction of the image and return top 5 [(name, score)]
    predictions = inference(IMAGE1_PATH)

    #compute sift
    runSIFT(image1, image2)

    # Color Histogram
    csvreader = csv.reader(open("colorhist.csv", "rb"))
    results = []

    test_files = glob.glob("test/*.jpg")
    for path in test_files: # this for loop only runs one time
        print path
        image = cv2.imread(path)
        for _ in xrange(1500):
            row = csvreader.next()
            filename = row[0]
            label = row[1]
            color_hist_train = np.array([float(col) for col in row[2:]])
            color_hist_test = color_descriptor.describe(image)
            results.append((label, calculate_distance(color_hist_test, color_hist_train)))

        results = sorted(results, key=lambda x: x[1])[-16:]
        print results
        break