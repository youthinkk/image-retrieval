"""
Template for Text Features
"""
from PIL import Image, ExifTags
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

#search for images associated to texts(training data)
def queryTrainText(query):
    train_list = processFile("../data/train_text_tags.txt")
    train_list_len = len(train_list)

    #dictionary of matched images
    matched = {}

    i = 0
    while (i < train_list_len): #loop through the entire tags texts

        j = 1
        single_list_len = len(train_list[i]) #loop through single line

        while (j < single_list_len):
            if (query.lower() in train_list[i][j].lower()) and (not matched.has_key(train_list[i][j])): #new key
                matched[train_list[i][j]] = [train_list[i][0].replace('.jpg','')] #make a list in the key
                j += 1
            elif (query.lower() in train_list[i][j].lower()) and (matched.has_key(train_list[i][j])): #existing key
                matched[train_list[i][j]].append(train_list[i][0].replace('.jpg',''))
                j += 1
            else:
                j += 1
        i += 1

    return matched

#search for images associated to texts(test data)
def queryTestText(query):
    test_list = processFile("../data/test_text_tags.txt")
    test_list_len = len(test_list)

    #dictionary of matched images
    matched = {}

    i = 0
    while (i < test_list_len): #loop through the entire tags texts

        j = 1
        single_list_len = len(test_list[i]) #loop through single line

        while (j < single_list_len):
            if (query.lower() in test_list[i][j].lower()) and (not matched.has_key(test_list[i][j])): #new key
                matched[test_list[i][j]] = [test_list[i][0].replace('.jpg','')] #make a list in the key
                j += 1
            elif (query.lower() in test_list[i][j].lower()) and (matched.has_key(test_list[i][j])): #existing key
                matched[test_list[i][j]].append(test_list[i][0].replace('.jpg',''))
                j += 1
            else:
                j += 1
        i += 1

    return matched

# #search of tags of image and
# def newImageQuery(image_path):
#     img = Image.open(image_path)
#
#     tag_name_to_id = dict([(v, k) for k, v in ExifTags.TAGS.items()])
#
#     exif_data = img._getexif()
#     exif = {
#         TAGS[k]: v
#         for k, v in img._getexif().items()
#         if k in TAGS
#         }
#     print exif_data
#     print exif
#     return 0