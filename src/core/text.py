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

#encode list of unicode to string
def toString(list):
    for i in range (len(list)):
        list[i] = str(list[i])
    return list

#search for images associated to texts(training data)
def singleQueryTrainText(query, matched, train_list):
    train_list_len = len(train_list)

    i = 0
    while (i < train_list_len): #loop through the entire tags texts

        j = 1
        single_list_len = len(train_list[i]) #loop through single line

        #adding matched images into dictionary under the key of the matched keyword
        while (j < single_list_len):
            if (query.lower() in train_list[i][j].lower()) and (not matched.has_key(train_list[i][0].replace('.jpg',''))): #new key
                matched[train_list[i][0].replace('.jpg','')] = 1 #make a list in the key
                j += 1
            elif (query.lower() in train_list[i][j].lower()) and (matched.has_key(train_list[i][0].replace('.jpg',''))): #existing key
                matched[train_list[i][0].replace('.jpg','')] += 1
                j += 1
            else:
                j += 1
        i += 1

    return matched

def topNlist(dict, N): #place N number of top search images in list of descending order
    count = 0
    list_out = []
    for w in sorted(dict, key=dict.get, reverse=True): #store up to top 20 most relevant images into out
        #print w, dict[w]
        list_out.append(w)
        count += 1
        if count >= N:
            break
    return list_out

#free text search
def freeTextSearch(string_query):
    train_list = processFile("../data/train_text_tags.txt")
    test_list = processFile("../data/test_text_tags.txt")

    syns = wordnet.synsets("program")

    s = set(stopwords.words('english')) #set for stemming
    input_query_list = filter(lambda w: not w in s, string_query.split()) #list of non stop words
    for i in range(len(input_query_list)): #stem and resolve all query words
        input_query_list[i] = stem(input_query_list[i])
    input_query_list = list(set(input_query_list)) #remove duplicates

    full_query_list = input_query_list #define a full query list which will include all synonym
    for i in range(len(input_query_list)):
        full_query_list = full_query_list + toString(synonym(input_query_list[i]))
    full_query_list = list(set(full_query_list)) #this is the COMPLETE set of queries w/o duplicates

    train_index = {} #initialize empty index to be updated
    test_index = {}
    print full_query_list
    for i in range(len(full_query_list)): #run through all the queries and return a dictionary
        train_index = singleQueryTrainText(full_query_list[i], train_index, train_list)
        test_index = singleQueryTrainText(full_query_list[i], test_index, test_list)

    #print index

    train_out = topNlist(train_index, 20)
    test_out = topNlist(test_index, 20)

    return train_out, test_out