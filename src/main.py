import cv2
from histogram import ColorDescriptor

COLOR_BINS = (8, 12, 3)

color_descriptor = ColorDescriptor(COLOR_BINS)

if __name__ == '__main__':
    image1 = cv2.imread("../data/train/data/balloons/0079_1387720721.jpg")
    image2 = cv2.imread("../data/train/data/balloons/0066_2283108985.jpg")

    color_similarity = color_descriptor.similarity(image1, image1)