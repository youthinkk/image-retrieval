import glob
import cv2
import pickle
from src.core import describe_color
from src.core import SIFTDescriptor, train_bow
from src.core import LearningDescriptor
from src.core import TagDescriptor

DATA_FOLDER = "../../data"
TRAIN_FOLDER = DATA_FOLDER + "/train"
TEST_FOLDER = DATA_FOLDER + "/test"
INDEX_FOLDER = DATA_FOLDER + "/index"

TRAIN_COLOR_INDEX_PATH = INDEX_FOLDER + "/train_histogram.csv"
TEST_COLOR_INDEX_PATH = INDEX_FOLDER + "/test_histogram.csv"

TRAIN_SIFT_INDEX_PATH = INDEX_FOLDER + "/train_sift.csv"
TEST_SIFT_INDEX_PATH = INDEX_FOLDER + "/test_sift.csv"
VISUAL_VOCABULARY_PATH = INDEX_FOLDER + "/visual_vocab"

TEST_LEARNING_INDEX_PATH = INDEX_FOLDER + "/test_learning.csv"

CATEGORY_TAGS = DATA_FOLDER + "/category_tags"
TEST_IMAGE_TAGS = DATA_FOLDER + "/test_text_tags.txt"


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


def generate_sift_index_file(folder_directory, output_path):
    file_paths = glob.glob(folder_directory + "/*.jpg")
    output = open(output_path, "w")
    sift_descriptor = SIFTDescriptor(VISUAL_VOCABULARY_PATH)

    for path in file_paths:
        image_id = path[path.rfind("/") + 1:]
        label = image_id.split("_")[-1][:-4]

        image = cv2.imread(path, 0)
        sift_feature = sift_descriptor.describe(image)
        sift_feature = [str(f) for f in sift_feature]
        output.write("%s,%s,%s\n" % (image_id, label, ",".join(sift_feature)))

    output.close()


def generate_learning_index_file(folder_directory, output_path):
    file_paths = glob.glob(folder_directory + "/*.jpg")
    output = open(output_path, "w")
    learning_descriptor = LearningDescriptor()

    for path in file_paths:
        image_id = path[path.rfind("/") + 1:]
        label = image_id.split("_")[-1][:-4]

        predictions = learning_descriptor.inference(path)
        predictions = [str(f) for f in predictions]
        output.write("%s,%s,%s\n" % (image_id, label, ",".join(predictions)))

    output.close()


def generate_text_index_file(folder_directory, output_path):
    file_paths = glob.glob(folder_directory + "/*.jpg")
    tag_descriptor = TagDescriptor(CATEGORY_TAGS)
    test_tags = {}
    test_score = {}

    f = open(TEST_IMAGE_TAGS)
    for _ in xrange(50):
        line = f.readline().split()
        test_tags[line[0]] = line[1:]

    for path in file_paths:
        image_id = path[path.rfind("/") + 1:]
        label = image_id.split("_")[-1][:-4]
        raw_id = image_id.replace("_" + label, "")
        if test_tags.has_key(raw_id):
            test_score[image_id] = tag_descriptor.get_score(test_tags.get(raw_id))

    fileObject = open(output_path, 'wb')
    pickle.dump(test_score, fileObject)
    fileObject.close()



# Generate BoW
# train_bow(1262)

# Generate index file for training images
# generate_color_index_file(TRAIN_FOLDER, TRAIN_COLOR_INDEX_PATH)
# generate_sift_index_file(TRAIN_FOLDER, TRAIN_SIFT_INDEX_PATH)

# Generate index file for testing images
# generate_color_index_file(TEST_FOLDER, TEST_COLOR_INDEX_PATH)
# generate_sift_index_file(TEST_FOLDER, TEST_SIFT_INDEX_PATH)
# generate_learning_index_file(TEST_FOLDER, TEST_LEARNING_INDEX_PATH)
generate_text_index_file(TEST_FOLDER, "../../data/index/test_tag_score")

