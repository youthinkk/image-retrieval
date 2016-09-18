import csv
import numpy as np

DATA_FOLDER = "../../data"
INDEX_FOLDER = DATA_FOLDER + "/index"

TRAIN_HISTOGRAM_INDEX_PATH = INDEX_FOLDER + "/train_histogram.csv"
TEST_HISTOGRAM_INDEX_PATH = INDEX_FOLDER + "/test_histogram.csv"


def get_train_histogram():
    return get_histogram_index_file(TRAIN_HISTOGRAM_INDEX_PATH)


def get_test_histogram():
    return get_histogram_index_file(TEST_HISTOGRAM_INDEX_PATH)


def get_histogram_index_file(file_path):
    reader = csv.reader(open(file_path, "rb"))
    dict = {}

    for row in reader:
        file_name = row[0]
        label = row[1]
        histogram = np.array([float(column) for column in row[2:]])

        dict[file_name] = (label, histogram)

    return dict
