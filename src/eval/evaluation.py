import numpy as np
from src.file import get_index
from src.core import color_similarity
from src.core import inference, learning_similarity

K_SIZE = 16

INDEX_FOLDER = "../../data/index"
TRAIN_COLOR_INDEX_PATH = INDEX_FOLDER + "/train_histogram.csv"
TEST_COLOR_INDEX_PATH = INDEX_FOLDER + "/test_histogram.csv"
TRAIN_SIFT_INDEX_PATH = INDEX_FOLDER + "/train_sift.csv"
TEST_SIFT_INDEX_PATH = INDEX_FOLDER + "/test_sift.csv"

# To test color only use [1, 0, 0]
# To test sift only use [0, 1, 0]
# To test deep learning only use [0, 0, 1]
WEIGHTS = np.array([1, 2, 3])


def avep(top_k, test_label):
    num_relevant = 0
    total = 0
    for i in xrange(K_SIZE):
        label = top_k[i][0]
        if label == test_label:
            num_relevant += 1
            total += num_relevant / i

    if num_relevant == 0:
        return 0

    return total / num_relevant


def calculate_score(similarities):
    return np.inner(WEIGHTS, similarities)


def eval():
    test_color_dict = get_index(TEST_COLOR_INDEX_PATH)
    train_color_dict = get_index(TRAIN_COLOR_INDEX_PATH)
    test__sift_dict = get_index(TEST_SIFT_INDEX_PATH)
    train_sift_dict = get_index(TRAIN_SIFT_INDEX_PATH)

    results = []
    total = 0

    for test_name, test_value in test_color_dict.iteritems():
        test_label = test_value[0]
        test_color = test_value[1]

        # Compute color feature of query image
        query_color = test_color

        # Compute sift feature
        query_sift = test__sift_dict.get(test_name)[1]

        # Compute deep learning predictions of query image
        predictions = inference("../../data/test/" + test_name)

        for file_name, train_value in train_color_dict.iteritems():
            train_label = train_value[0]
            train_color = train_value[1]

            # Compute similarity of different features
            color_sim = color_similarity(query_color, train_color)
            sift_sim = color_similarity(query_sift, train_sift_dict.get(file_name)[1])
            learning_sim = learning_similarity(predictions, train_label)

            score = calculate_score([color_sim, sift_sim, learning_sim])
            results.append((score, file_name))

        top_k = sorted(results, key=lambda x: x[0], reverse=True)[:K_SIZE]
        total += avep(top_k, test_label)
        break

    print "MAP: %.6f" % (total / 300)





# Color histogram accuracy
# Using Bhattacharyya distance: 0.0329166666667
# Using chi2 distance: 0.0358333333333
# print "Color histogram accuracy: ", eval_color()

eval()






#################################################################
# SIFT + SVM
#################################################################

# def train_svm():
#     f = open('../../data/category_names.txt', 'r')
#     categories = [item.strip('\r\n') for item in list(f)]
#     f.close()
#
#     train_desc = []
#     train_labels = []
#
#     train_dict = get_index(TRAIN_SIFT_INDEX_PATH)
#
#     for train_name, train_value in train_dict.iteritems():
#         train_label = train_value[0]
#         train_sift = train_value[1].astype("float32")
#         train_desc.append(np.array(train_sift))
#         train_labels.append(categories.index(train_label))
#
#     print len(train_desc)
#     print len(train_labels)
#     svm = cv2.SVM()
#     svm.train(np.array(train_desc), np.array(train_labels))
#     return svm

# svm = train_svm()
# test_dict = get_color_index(TEST_SIFT_INDEX_PATH)
# score = 0
# f = open('../../data/category_names.txt', 'r')
# categories = [item.strip('\r\n') for item in list(f)]
# f.close()
# for test_name, test_value in test_dict.iteritems():
#     test_label = test_value[0]
#     test_color = test_value[1]
#     prediction = svm.predict(test_color.astype("float32"))
#     if categories.index(test_label) == prediction:
#         score += 1
#
# print score
