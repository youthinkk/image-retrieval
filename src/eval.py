# import cv2
# import glob
# import os
#
#
# f = open('../data/category_names.txt', 'r')
# categories = [item.strip('\r\n') for item in list(f)]
# f.close()
#
# for cat in categories:
#     print cat
#     files = glob.glob("../data/train/data/%s/*.jpg" % cat)
#
#     for path in files:
#         os.rename(path, "train/" + path.split("/")[-1][0:-4] + "_%s.jpg" % cat)