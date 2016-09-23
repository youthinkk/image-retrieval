"""
Template for Text Features
"""
from stemming.porter2 import stem
from nltk.corpus import stopwords, wordnet

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

#returns a list of synonyms of the word
def synonym(word):
    synonyms = list()
    for syn in wordnet.synsets(word):
        for l in syn.lemmas():
            synonyms.append(l.name())
    synonyms = list((set(synonyms)))
    return synonyms

#search for images associated to texts(training data)
def singleQueryTrainText(query):
    train_list = processFile("../data/train_text_tags.txt")
    train_list_len = len(train_list)

    #dictionary of matched images
    matched = {}

    i = 0
    while (i < train_list_len): #loop through the entire tags texts

        j = 1
        single_list_len = len(train_list[i]) #loop through single line

        #adding matched images into dictionary under the key of the matched keyword
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
def singleQueryTestText(query):
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

#free text search
def freeTextSearch(string_query):
    syns = wordnet.synsets("program")

    s = set(stopwords.words('english')) #set for stemming
    input_query_list = filter(lambda w: not w in s, string_query.split()) #list of non stop words
    for i in range(len(input_query_list)): #stem and resolve all query words
        input_query_list[i] = stem(input_query_list[i])
    input_query_list = list(set(input_query_list)) #remove duplicates

    full_query_list = input_query_list #define a full query list which will include all synonym
    for i in range(len(input_query_list)):
        full_query_list = full_query_list + synonym(input_query_list[i])
    full_query_list = list(set(full_query_list)) #this is the COMPLETE set of queries

    #for i in range(len(query_list)): #concat all dictionay to make one big dict


    testing1 = singleQueryTrainText(input_query_list[0])
    testing2 = singleQueryTrainText(input_query_list[3])
    testing1.update(testing2)
    #print testing1

    return string_query