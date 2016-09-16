# USAGE
# python index.py --dataset dataset --index index.csv

from histogram import ColorDescriptor
import argparse
import glob
import cv2

# construct the argument parser and parse the arguments
ap = argparse.ArgumentParser()
ap.add_argument("-d", "--dataset", required=False, default='../data/train',
                help="Path to the directory that contains the images to be indexed")
ap.add_argument("-i", "--index", required=False, default='colorhist.csv',
                help="Path to where the computed index will be stored")
args = vars(ap.parse_args())

# initialize the color descriptor
color_descriptor = ColorDescriptor((8, 12, 3))

# open the output index file for writing
output = open(args["index"], "w")

# use glob to grab the image paths and loop over them
for image_path in glob.glob(args["dataset"] + "/*.jpg"):
    # extract the image ID (i.e. the unique filename) from the image
    # path and load the image itself
    image_id = image_path[image_path.rfind("/") + 1:]
    image = cv2.imread(image_path)

    # describe the image
    features = color_descriptor.describe(image)

    # write the features to file
    features = [str(f) for f in features]
    label = image_id.split("_")[-1][:-4]
    output.write("%s,%s,%s\n" % (image_id, label, ",".join(features)))

# close the index file
output.close()
