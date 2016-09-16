import cv2
from histogram import ColorDescriptor
from learning import inference
from keywords import runSIFT

COLOR_BINS = (8, 12, 3)

color_descriptor = ColorDescriptor(COLOR_BINS)

if __name__ == '__main__':
    IMAGE1_PATH = "../data/train/data/balloons/0079_1387720721.jpg"
    IMAGE2_PATH = "../data/train/data/balloons/0066_2283108985.jpg"
    image1 = cv2.imread(IMAGE1_PATH)
    image2 = cv2.imread(IMAGE2_PATH)

    # Compute color similarity and return a scalar similarity score
    color_similarity = color_descriptor.similarity(image1, image1)

    # Compute prediction of the image and return top 5 [(name, score)]
    predictions = inference(IMAGE1_PATH)

    #compute sift
    runSIFT(image1, image2)