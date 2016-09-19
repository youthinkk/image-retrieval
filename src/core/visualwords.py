import cv2
import glob
import pickle

##############################################################
# Uncomment the following to generate the visual vocab
##############################################################
# dictionarySize = 10      # NEED TO CHANGE THIS...
# sift = cv2.SIFT()
# BOW = cv2.BOWKMeansTrainer(dictionarySize)
#
# for p in glob.glob("../data/train/*.jpg"):
#     image = cv2.imread(p)
#     gray = cv2.cvtColor(image, cv2.CV_LOAD_IMAGE_GRAYSCALE)
#     kp, dsc= sift.detectAndCompute(gray, None)
#     BOW.add(dsc)
#
# #dictionary creation
# print "Start of clustering"
# dictionary = BOW.cluster()
# print "End of clustering"
# fileObject = open("../data/index/visual_vocab",'wb')
# pickle.dump(dictionary, fileObject)
# fileObject.close()

sift = cv2.SIFT()
fileObject = open("../data/index/visual_vocab", 'r')
dictionary = pickle.load(fileObject)

sift2 = cv2.DescriptorExtractor_create("SIFT")
bowDiction = cv2.BOWImgDescriptorExtractor(sift2, cv2.BFMatcher(cv2.NORM_L2))
bowDiction.setVocabulary(dictionary)


def generate_sift_index_file(folder_directory, output_path):
    file_paths = glob.glob(folder_directory + "/*.jpg")
    output = open(output_path, "w")

    for path in file_paths:
        image_id = path[path.rfind("/") + 1:]
        label = image_id.split("_")[-1][:-4]

        image = cv2.imread(path)
        sift_feature = bowDiction.compute(image, sift.detect(image))
        sift_feature = [str(f) for f in sift_feature[0]]
        output.write("%s,%s,%s\n" % (image_id, label, ",".join(sift_feature)))

    output.close()


# Generate histogram index file for training images
generate_sift_index_file("../data/train", "train_sift.csv")

# Generate histogram index file for testing images
generate_sift_index_file("../data/test", "test_sift.csv")