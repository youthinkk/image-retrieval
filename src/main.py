import cv2
from histogram import histogram_similarity


if __name__ == '__main__':
    image1 = cv2.imread("../data/train/data/balloons/0079_1387720721.jpg")
    image2 = cv2.imread("../data/train/data/balloons/0066_2283108985.jpg")

    histogram_similarity = histogram_similarity(image1, image2)