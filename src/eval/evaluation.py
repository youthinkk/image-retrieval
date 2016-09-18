from src.file import get_color_index
from src.core import color_similarity

K_SIZE = 16

INDEX_FOLDER = "../../data/index"
TRAIN_COLOR_INDEX_PATH = INDEX_FOLDER + "/train_histogram.csv"
TEST_COLOR_INDEX_PATH = INDEX_FOLDER + "/test_histogram.csv"


def local_accuracy(top_k, test_label):
    local_sum = 0.
    for i in xrange(K_SIZE):
        label = top_k[i][0]
        if label == test_label:
            local_sum += 1

    return local_sum / K_SIZE


def eval_color():
    test_dict = get_color_index(TEST_COLOR_INDEX_PATH)
    train_dict = get_color_index(TRAIN_COLOR_INDEX_PATH)

    test_size = len(test_dict)
    accuracy_sum = 0.
    results = []

    for test_name, test_value in test_dict.iteritems():
        test_label = test_value[0]
        test_color = test_value[1]

        for train_name, train_value in train_dict.iteritems():
            train_label = train_value[0]
            train_color = train_value[1]
            similarity = color_similarity(test_color, train_color)
            results.append((train_label, similarity))

        top_k = sorted(results, key=lambda x: x[1], reverse=True)[:K_SIZE]
        accuracy_sum += local_accuracy(top_k, test_label)

    accuracy = accuracy_sum / test_size

    return accuracy


# Color histogram accuracy
# Using Bhattacharyya distance: 0.0329166666667
# Using chi2 distance: 0.0358333333333
print "Color histogram accuracy: ", eval_color()
