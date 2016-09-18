import cv2
import numpy as np
from src.core import describe_color, color_similarity
from src.core import inference, learning_similarity
from src.file import get_color_index

K_SIZE = 16
TRAIN_COLOR_INDEX_PATH = "../data/index/train_histogram.csv"


def calculate_socre(similarities):
    weights = np.array([0, 2])
    return np.inner(weights, similarities)


if __name__ == '__main__':
    query_path = "../data/test/0001_127194972_balloons.jpg"
    query_image = cv2.imread(query_path)
    train_color_dict = get_color_index(TRAIN_COLOR_INDEX_PATH)
    results = []

    # Compute color feature of query image
    query_color = describe_color(query_image)

    # Compute deep learning predictions of query image
    predictions = inference(query_path)

    for file_name, train_value in train_color_dict.iteritems():
        train_label = train_value[0]
        train_color = train_value[1]

        color_sim = color_similarity(query_color, train_color)
        learning_sim = learning_similarity(predictions, train_label)

        score = calculate_socre([color_sim, learning_sim])
        results.append((file_name, score))

    top_k = sorted(results, key=lambda x: x[1], reverse=True)[:K_SIZE]
    print top_k

    '''
    #compute sift
    runSIFT(image1, image2)

    '''