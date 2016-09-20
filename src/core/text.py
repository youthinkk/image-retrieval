"""
Template for Text Features
"""
from PIL.ExifTags import TAGS
from PIL.ExifTags import TAGS

#Specify file path of training and test text tags
train_Tags = "../data/train_text_tags.txt"
test_Tags = "../data/test_text_tags.txt"


#read a single line in the data and process them into a returned list
def processLine(input):
    list = input.split()
    return list

#find the file of specified name and return a list of keywords in each line
def processFile(path):
    l = list()
    with open(path) as f:
        count = 0
        for line in f:
            linelist = processLine(line)
            l.insert(count, linelist)
            count += 1
    return l

#search for images associated to texts
def queryTrainText(query):
    train_list = processFile("../data/train_text_tags.txt")
    train_list_len = len(train_list)

    #list of matched images
    matched = list()
    match_count = 0

    i = 0
    while (i < train_list_len):

        j = 1
        single_list_len = len(train_list[i])

        while (j < single_list_len):
            if query.lower() == train_list[i][j].lower():
                matched.insert(match_count, train_list[i][0])
                match_count += 1
                break
            else:
                j += 1
        i += 1

    return matched

#search for images associated to texts
def queryTestText(query):
    test_list = processFile("../data/test_text_tags.txt")
    test_list_len = len(test_list)

    #list of matched images
    matched = list()
    match_count = 0

    i = 0
    while (i < test_list_len):

        j = 1
        single_list_len = len(test_list[i])

        while (j < single_list_len):
            if query.lower() == test_list[i][j].lower():
                matched.insert(match_count, test_list[i][0])
                match_count += 1
                break
            else:
                j += 1
        i += 1

    return matched