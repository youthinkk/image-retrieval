"""
Template for Text Features
"""
from PIL.ExifTags import TAGS
from PIL.ExifTags import TAGS

#read a single line in the data and process them into a returned list
def processLine(input):
    arr = input.split()
    return arr

#find the file of specified name and return path
def findFile(name):
    path = "../data/test/" + name
    return path

#search for image and imput tags into image description
def searchInject(list):
    fileName = list[0]
    file = findFile(fileName)
    list.pop(0)
    tags = list
    for i in tags:
