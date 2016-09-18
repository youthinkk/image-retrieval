import cv2
from src.core import describe_color, color_similarity
from src.core import inference, learning_similarity


if __name__ == '__main__':
    IMAGE1_PATH = "../data/train/0090_2078259391_zebra.jpg"
    IMAGE2_PATH = "../data/train/0020_386347744_antlers.jpg"
    image1 = cv2.imread(IMAGE1_PATH)
    image2 = cv2.imread(IMAGE2_PATH)

    # Compute color similarity
    histogram1 = describe_color().describe(image1)
    histogram2 = describe_color().describe(image2)
    color_similarity = color_similarity(histogram1, histogram2)

    # Compute deep learning similarity
    predictions = inference(IMAGE1_PATH)
    learning_similarity = learning_similarity(predictions, "zebra")

    print learning_similarity

    '''
    #compute sift
    runSIFT(image1, image2)

    '''