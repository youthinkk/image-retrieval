import csv
import numpy as np


def get_color_index(file_path):
    reader = csv.reader(open(file_path, "rb"))
    dict = {}

    for row in reader:
        file_name = row[0]
        label = row[1]
        color = np.array([float(column) for column in row[2:]])

        dict[file_name] = (label, color)

    return dict
