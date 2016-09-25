import cv2
import numpy as np
from src.core import describe_color
from src.core import LearningDescriptor, learning_similarity
from src.core import SIFTDescriptor
from src.core import TagDescriptor
from src.core import color_similarity, sift_similarity, remove_duplicate
from src.file import get_index

K_SIZE = 16
TRAIN_COLOR_INDEX_PATH = "../data/index/train_histogram.csv"
TRAIN_SIFT_INDEX_PATH = "../data/index/train_sift.csv"
VISUAL_VOCABULARY_PATH = "../data/index/visual_vocab"
CATEGORY_TAGS_PATH = "../data/category_tags"
DEFAULT_WEIGHTS = np.array([0.84757445, 3.67971924, 48.4926241, 1])


class Searcher:

    def __init__(self, weights=DEFAULT_WEIGHTS):
        self.weights = weights
        self.sift_descriptor = SIFTDescriptor(VISUAL_VOCABULARY_PATH)
        self.learning_descriptor = LearningDescriptor()
        self.tag_descriptor = TagDescriptor(CATEGORY_TAGS_PATH)

        # Load training data
        train_color_list = get_index(TRAIN_COLOR_INDEX_PATH).items()
        train_sift_dict = get_index(TRAIN_SIFT_INDEX_PATH)
        self.train_length = len(train_color_list)
        self.train_file_names = [train_color_list[i][0] for i in xrange(self.train_length)]
        self.train_labels = [train_color_list[i][1][0] for i in xrange(self.train_length)]
        self.train_colors = [train_color_list[i][1][1] for i in xrange(self.train_length)]
        self.train_sifts = [train_sift_dict.get(self.train_file_names[i])[1] for i in xrange(self.train_length)]

    def retrieve_images(self, query_path, tags):
        query_image = cv2.imread(query_path)
        results = []

        # Compute color feature of query image
        query_color = describe_color(query_image)

        # Compute sift feature
        query_sift = self.sift_descriptor.describe(cv2.cvtColor(query_image, 0))

        # Compute deep learning predictions of query image
        predictions = self.learning_descriptor.inference(query_path)

        tags_score, matched_images = self.tag_descriptor.get_score(tags)

        for i in xrange(self.train_length):
            color_sim = color_similarity(query_color, self.train_colors[i])
            sift_sim = sift_similarity(query_sift, self.train_sifts[i])
            learning_sim = learning_similarity(predictions, self.train_labels[i])

            mult = 1
            if self.train_file_names[i] in matched_images:
                mult = 2

            score = self.calculate_score([color_sim, sift_sim, learning_sim, mult * tags_score[self.train_labels[i]]])
            results.append((score, self.train_file_names[i]))

        results = sorted(results, key=lambda x: x[0], reverse=True)
        results = remove_duplicate(results)
        top_k = results[:K_SIZE]
        return top_k

    def calculate_score(self, similarities):
        return np.inner(self.weights, similarities)

    def set_weights(self, weights):
        self.weights = weights

    def get_weights(self):
        return self.weights

    def set_tag_score(self, score):
        self.tag_score = score