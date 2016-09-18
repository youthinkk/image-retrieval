import glob
import cv2
from src.core import describe_color

DATA_FOLDER = "../data"
TRAIN_FOLDER = DATA_FOLDER + "/train"
TEST_FOLDER = DATA_FOLDER + "/test"
INDEX_FOLDER = DATA_FOLDER + "/index"

TRAIN_COLOR_INDEX_PATH = INDEX_FOLDER + "/train_histogram.csv"
TEST_COLOR_INDEX_PATH = INDEX_FOLDER + "/test_histogram.csv"


def generate_color_index_file(folder_directory, output_path):
    file_paths = glob.glob(folder_directory + "/*.jpg")
    output = open(output_path, "w")

    for path in file_paths:
        image_id = path[path.rfind("/") + 1:]
        label = image_id.split("_")[-1][:-4]

        image = cv2.imread(path)
        color = describe_color(image)
        color = [str(f) for f in color]
        output.write("%s,%s,%s\n" % (image_id, label, ",".join(color)))

    output.close()


# Generate histogram index file for training images
generate_color_index_file(TRAIN_FOLDER, TRAIN_COLOR_INDEX_PATH)

# Generate histogram index file for testing images
generate_color_index_file(TEST_FOLDER, TEST_COLOR_INDEX_PATH)
