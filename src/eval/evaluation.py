from src.file import get_color_index
from src.core import color_similarity

K_SIZE = 16

DATA_FOLDER = "../../data"
INDEX_FOLDER = DATA_FOLDER + "/index"
TRAIN_COLOR_INDEX_PATH = INDEX_FOLDER + "/train_histogram.csv"
TEST_COLOR_INDEX_PATH = INDEX_FOLDER + "/test_histogram.csv"


def eval_histogram():
    test_dict = get_color_index(TEST_COLOR_INDEX_PATH)
    train_dict = get_color_index(TRAIN_COLOR_INDEX_PATH)
    results = []

    for test_name, test_value in test_dict.iteritems():
        test_histogram = test_value[1]

        for train_name, train_value in train_dict.iteritems():
            train_label = train_value[0]
            train_histogram = train_value[1]
            similarity = color_similarity(test_histogram, train_histogram)
            results.append((train_label, similarity))

        top_k = sorted(results, key=lambda x: x[1], reverse=True)[:K_SIZE]
        print top_k
        break

eval_histogram()
