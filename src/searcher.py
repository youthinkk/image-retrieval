import cv2
import numpy as np
from src.core import describe_color
from src.core import LearningDescriptor, learning_similarity
from src.core import SIFTDescriptor
from src.core import compute_similarity
from src.file import get_index

K_SIZE = 16
TRAIN_COLOR_INDEX_PATH = "../data/index/train_histogram.csv"
TRAIN_SIFT_INDEX_PATH = "../data/index/train_sift.csv"
VISUAL_VOCABULARY_PATH = "../data/index/visual_vocab"
WEIGHTS = np.array([1, 2, 3])


class Searcher:

    def __init__(self, weights=WEIGHTS):
        self.k_size = K_SIZE
        self.weights = weights
        self.train_color_list = get_index(TRAIN_COLOR_INDEX_PATH).items()
        self.train_sift_dict = get_index(TRAIN_SIFT_INDEX_PATH)
        self.sift_descriptor = SIFTDescriptor(VISUAL_VOCABULARY_PATH)
        self.learning_descriptor = LearningDescriptor()

        self.train_length = len(self.train_color_list)
        self.train_file_names = [self.train_color_list[i][0] for i in xrange(self.train_length)]
        self.train_labels = [self.train_color_list[i][1][0] for i in xrange(self.train_length)]
        self.train_colors = [self.train_color_list[i][1][1] for i in xrange(self.train_length)]
        self.train_sifts = [self.train_sift_dict.get(self.train_file_names[i])[1] for i in xrange(self.train_length)]

    def retrieve_images(self, query_path):
        query_image = cv2.imread(query_path)
        results = []

        # Compute color feature of query image
        query_color = describe_color(query_image)

        # Compute sift feature
        query_sift = self.sift_descriptor.describe(cv2.cvtColor(query_image, 0))

        # Compute deep learning predictions of query image
        predictions = self.learning_descriptor.inference(query_path)

        for i in xrange(self.train_length):
            color_sim = compute_similarity(query_color, self.train_colors[i])
            sift_sim = compute_similarity(query_sift, self.train_sifts[i])
            learning_sim = learning_similarity(predictions, self.train_labels[i])

            score = self.calculate_score([color_sim, sift_sim, learning_sim])
            results.append((score, self.train_file_names[i]))

        top_k = sorted(results, key=lambda x: x[0], reverse=True)[:K_SIZE]
        return top_k

    def calculate_score(self, similarities):
        return np.inner(self.weights, similarities)
