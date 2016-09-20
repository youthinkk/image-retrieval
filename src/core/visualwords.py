import cv2
import glob
import pickle


class SIFTDescriptor:

    def __init__(self, vocab_path):
        file_object = open(vocab_path, 'r')
        vocab = pickle.load(file_object)

        sift2 = cv2.DescriptorExtractor_create("SIFT")
        bow_dictionary = cv2.BOWImgDescriptorExtractor(sift2, cv2.BFMatcher(cv2.NORM_L2))
        bow_dictionary.setVocabulary(vocab)

        self.bow_dictionary = bow_dictionary
        self.sift = cv2.SIFT()

    def describe(self, img):
        return self.bow_dictionary.compute(img, self.sift.detect(img))[0]


def train_bow(dictionary_size):
    sift = cv2.SIFT()
    BOW = cv2.BOWKMeansTrainer(dictionary_size)

    for p in glob.glob("../data/train/*.jpg"):
        image = cv2.imread(p, 0)
        kp, dsc = sift.detectAndCompute(image, None)
        BOW.add(dsc)

    # dictionary creation
    print "Start of clustering"
    dictionary = BOW.cluster()
    print "End of clustering"
    fileObject = open("../data/index/visual_vocab", 'wb')
    pickle.dump(dictionary, fileObject)
    fileObject.close()
