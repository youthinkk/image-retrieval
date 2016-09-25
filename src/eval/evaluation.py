import random
import numpy as np
import pickle
from src.file import get_index
from src.core import learning_similarity
from src.core import color_similarity, sift_similarity, remove_duplicate

K_SIZE = 16

TEST_FOLDER = "../../data/test/"
INDEX_FOLDER = "../../data/index"
TRAIN_COLOR_INDEX_PATH = INDEX_FOLDER + "/train_histogram.csv"
TEST_COLOR_INDEX_PATH = INDEX_FOLDER + "/test_histogram.csv"
TRAIN_SIFT_INDEX_PATH = INDEX_FOLDER + "/train_sift.csv"
TEST_SIFT_INDEX_PATH = INDEX_FOLDER + "/test_sift.csv"
TEST_LEARNING_INDEX_PATH = INDEX_FOLDER + "/test_learning.csv"
TEST_TAG_SCORE_PATH = INDEX_FOLDER + "/test_tag_score"
WEIGHTS_INDEX_PATH = "../../data/weights.csv"

WEIGHTS = np.array([7.87123598, 7.29097985, 33.38541097, 1])
WEIGHTS_TRAINING_SIZE = 1000


def average_precision(top_k, truth_label):
    num_relevant = 0.
    total_precision = 0.
    for i in xrange(K_SIZE):
        label = top_k[i][1]
        if label == truth_label:
            num_relevant += 1
            total_precision += num_relevant / (i + 1)

    if num_relevant == 0:
        return 0

    return total_precision / num_relevant


def calculate_score(weights, similarities):
    return np.inner(weights, similarities)


class Evaluation:

    def __init__(self):
        # Load training data
        self.train_color_dict = get_index(TRAIN_COLOR_INDEX_PATH)
        self.train_sift_dict = get_index(TRAIN_SIFT_INDEX_PATH)

        # Load testing data
        self.test_color_dict = get_index(TEST_COLOR_INDEX_PATH)
        self.test_sift_dict = get_index(TEST_SIFT_INDEX_PATH)
        self.test_learning_dict = get_index(TEST_LEARNING_INDEX_PATH)
        self.test_tag_dict = pickle.load(open(TEST_TAG_SCORE_PATH, "r"))

    def run(self, weights):
        total = 0.

        test_size = len(self.test_color_dict)
        for test_name, test_value in self.test_color_dict.iteritems():
            results = []
            test_label = test_value[0]
            test_color = test_value[1]

            # Compute color feature of query image
            query_color = test_color

            # Compute sift feature
            query_sift = self.test_sift_dict.get(test_name)[1]

            # Compute deep learning predictions of query image
            predictions = self.test_learning_dict.get(test_name)[1]

            # Get how well the tags score in each category
            category_score = self.test_tag_dict.get(test_name)

            for file_name, train_value in self.train_color_dict.iteritems():
                train_label = train_value[0]
                train_color = train_value[1]

                # Compute similarity of different features
                color_sim = color_similarity(query_color, train_color)
                sift_sim = sift_similarity(query_sift, self.train_sift_dict.get(file_name)[1])
                learning_sim = learning_similarity(predictions, train_label)

                if category_score is None:
                    tag_score = 0
                else:
                    tag_score = category_score.get(train_label)

                score = calculate_score(weights, [color_sim, sift_sim, learning_sim, tag_score])
                results.append((score, train_label))

            results = sorted(results, key=lambda x: x[0], reverse=True)
            results = remove_duplicate(results)
            top_k = results[:K_SIZE]
            total += average_precision(top_k, test_label)

        return "%.10f" % (total / test_size)


evaluation = Evaluation()

# Color histogram accuracy
# print "Color histogram: ", evaluation.run([1, 0, 0, 0])

# SIFT accuracy
# print "SIFT: ", evaluation.run([0, 1, 0, 0])

# Deep learning accuracy
# print "Deep learning: ", evaluation.run([0, 0, 1, 0])

# Text accuracy
# print "Text: ", evaluation.run([0, 0, 0, 1])

# Overall accuracy
# print "Overall: ", evaluation.run(WEIGHTS)

# Find most optimal accuracy
# for i in xrange(WEIGHTS_TRAINING_SIZE):
#     w = []
#     for _ in xrange(3):     # Number of features
#         w.append(random.uniform(0.1, 50.0))
#
#     MAP = evaluation.run(w)
#
#     with open(WEIGHTS_INDEX_PATH, "a") as weight_path:
#         weight_path.write("%s,%s\n" % (str(np.array(w)), str(MAP)))
#     print "Iteration ", (i+1), ": ", MAP


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
