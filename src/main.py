import cv2
from src.main.histogram import ColorDescriptor, color_similarity

from src.core.learning import inference, learning_similarity

COLOR_BINS = (8, 12, 3)

color_descriptor = ColorDescriptor(COLOR_BINS)

if __name__ == '__main__':
    IMAGE1_PATH = "../data/train/0090_2078259391_zebra.jpg"
    IMAGE2_PATH = "../data/train/0020_386347744_antlers.jpg"
    image1 = cv2.imread(IMAGE1_PATH)
    image2 = cv2.imread(IMAGE2_PATH)

    # Compute color similarity
    histogram1 = color_descriptor.describe(image1)
    histogram2 = color_descriptor.describe(image2)
    color_similarity = color_similarity(histogram1, histogram2)

    # Compute deep learning similarity
    predictions = inference(IMAGE1_PATH)
    learning_similarity = learning_similarity(predictions, "zebra")

    print learning_similarity

    '''
    #compute sift
    runSIFT(image1, image2)

    '''