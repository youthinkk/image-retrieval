"""
Template for Visual Keyword
"""
import os
import cv2

# creeate SIFT object
detector = cv2.SIFT()

#identify OS to run SIFT
def process_image(imagename, resultname):
    """ process an image and save the results in a .key ascii file"""
    # check if linux or windows
    if os.name == "nt":
        cmd = "siftWin32 <" + imagename + ">" + resultname
    else:
        cmd = "./sift <" + imagename + ">" + resultname
    os.system(cmd)
    print 'processed', imagename

#***************************************************************************

#generate keypoints for the image
def findKp(img):
    return detector.detect(img)

def runSIFT(image1, image2):
    #gray out both imgs
    gray1 = cv2.cvtColor(image1,cv2.COLOR_BGR2GRAY)
    gray2 = cv2.cvtColor(image2,cv2.COLOR_BGR2GRAY)

    #finding the keypoints for each individual image
    kp1 = findKp(gray1)
    kp2 = findKp(gray2)

    #draw kp on img (not necessary)
    newImg1 = cv2.drawKeypoints(gray1, kp1)
    newImg2 = cv2.drawKeypoints(gray2, kp2)
    # cv2.imwrite('testOriginal.jpg', gray1)
    # cv2.imwrite('test.jpg', newImg1)

    #comparing the keypoints of the 2 images



    return 0