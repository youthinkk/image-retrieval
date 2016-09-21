import cv2
import numpy as np
from src.core import describe_color, color_similarity
from src.core import inference, learning_similarity
from src.core import SIFTDescriptor
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
        self.train_color_dict = get_index(TRAIN_COLOR_INDEX_PATH)
        self.train_sift_dict = get_index(TRAIN_SIFT_INDEX_PATH)
        self.sift_descriptor = SIFTDescriptor(VISUAL_VOCABULARY_PATH)

    def calculate_score(self, similarities):
        return np.inner(self.weights, similarities)

    def retrieve_images(self, query_path):
        query_image = cv2.imread(query_path)
        results = []

        # Compute color feature of query image
        query_color = describe_color(query_image)

        # Compute sift feature
        query_sift = self.sift_descriptor.describe(cv2.cvtColor(query_image, 0))

        # Compute deep learning predictions of query image
        predictions = inference(query_path)

        for file_name, train_value in self.train_color_dict.iteritems():
            train_label = train_value[0]
            train_color = train_value[1]

            # Compute similarity of different features
            color_sim = color_similarity(query_color, train_color)
            sift_sim = color_similarity(query_sift, self.train_sift_dict.get(file_name)[1])
            learning_sim = learning_similarity(predictions, train_label)

            score = self.calculate_score([color_sim, sift_sim, learning_sim])
            results.append((score, file_name))

        top_k = sorted(results, key=lambda x: x[0], reverse=True)[:K_SIZE]
        return top_k
