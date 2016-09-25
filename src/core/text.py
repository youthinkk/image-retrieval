"""
Template for Text Features
"""
import pickle
import glob
from stemming.porter2 import stem
from collections import Counter


class TagDescriptor:

    def __init__(self, category_tags):
        f = open(category_tags, 'rb')
        self.cat_dict = pickle.load(f)
        f.close()
        self.categories = ['alley', 'antlers', 'baby', 'balloons', 'beach', 'bear', 'birds', 'boats', 'cars', 'cat',
                           'computer', 'coral', 'dog', 'fish', 'flags', 'flowers', 'horses', 'leaf', 'plane', 'rainbow',
                           'rocks', 'sign', 'snow', 'tiger', 'tower', 'train', 'tree', 'whales', 'window', 'zebra']

    @staticmethod
    def filter_tags(tags):
        filtered = []
        for word in tags:
            if len(word) > 2 and word.isalpha():
                filtered.append(stem(word).lower())

        return filtered

    # returns a score of how well the tags match each category
    def get_score(self, tags):
        if tags is None:
            tags = []
        tags = self.filter_tags(tags)

        score = {}
        matched_images = []
        for cat in self.categories:
            cat_tags = self.cat_dict.get(cat)
            matched = []
            if cat_tags is None:
                score[cat] = 0
            else:
                total = 0
                for tag in tags:
                    if cat_tags.has_key(tag):
                        total += cat_tags.get(tag)[0]
                        matched.append(cat_tags.get(tag)[1])

                score[cat] = total

            if len(matched) > 0:
                matched_images += list(set.intersection(*matched))

        return score, matched_images


#####################################################################
# Code to generate the category tags
#####################################################################

# dictionary = {}
#
# f = open("../../data/train_text_tags.txt")
# for _ in xrange(848):
#     line = f.readline().split()
#     dictionary[line[0]] = TagDescriptor.filter_tags(line[1:])
#
# TRAIN_FOLDER = "../../data/train2/*"      # Folder where all the training images are grouped by category
# folders = glob.glob(TRAIN_FOLDER)
# category_tags = {}
# for fo in folders:
#     images = glob.glob(fo + "/*.jpg")
#     cat = fo.split("/")[-1]
#     tags = []
#     for img in images:
#         filename = img.split("/")[-1]
#         if dictionary.has_key(filename):
#             tags += (dictionary.get(filename))
#
#     num_of_tags = float(len(tags))
#     weights = {}
#     for tag, count in Counter(tags).iteritems():
#         occurance = set()
#         for img in images:
#             filename = img.split("/")[-1]
#             if dictionary.has_key(filename):
#                 if tag in dictionary.get(filename):
#                     occurance.add(filename.replace(".jpg", "_%s.jpg" % cat))
#         weights[tag] = (count / num_of_tags, occurance)
#     category_tags[cat] = weights
#
# print category_tags
#
# fileObject = open("../../data/category_tags", 'wb')
# pickle.dump(category_tags, fileObject)
# fileObject.close()