from histogram import ColorDescriptor
import glob
import cv2

DATA_FOLDER = "../data"
TRAIN_FOLDER = DATA_FOLDER + "/train"
TEST_FOLDER = DATA_FOLDER + "/test"
INDEX_FOLDER = DATA_FOLDER + "/index"

TRAIN_HISTOGRAM_INDEX_PATH = INDEX_FOLDER + "/train_histogram.csv"
TEST_HISTOGRAM_INDEX_PATH = INDEX_FOLDER + "/test_histogram.csv"


def generate_histogram_index_file(folder_directory, output_path):
    color_descriptor = ColorDescriptor((8, 12, 3))
    file_paths = glob.glob(folder_directory + "/*.jpg")
    output = open(output_path, "w")

    for path in file_paths:
        image_id = path[path.rfind("/") + 1:]
        label = image_id.split("_")[-1][:-4]

        image = cv2.imread(path)
        histogram = color_descriptor.describe(image)
        histogram = [str(f) for f in histogram]
        output.write("%s,%s,%s\n" % (image_id, label, ",".join(histogram)))

    output.close()


generate_histogram_index_file(TRAIN_FOLDER, TRAIN_HISTOGRAM_INDEX_PATH)
generate_histogram_index_file(TEST_FOLDER, TEST_HISTOGRAM_INDEX_PATH)
