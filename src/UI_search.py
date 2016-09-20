import cv2
from Tkinter import *
import tkFileDialog
from PIL import Image, ImageTk
import numpy as np
from src.core import describe_color, color_similarity
from src.core import inference, learning_similarity
from src.core import SIFTDescriptor
from src.file import get_index

K_SIZE = 16
TRAIN_COLOR_INDEX_PATH = "../data/index/train_histogram.csv"
TRAIN_SIFT_INDEX_PATH = "../data/index/train_sift.csv"
VISUAL_VOCABULARY_PATH = "../data/index/visual_vocab"


class UI_class:
    def __init__(self, master, search_path):
        self.search_path = search_path
        self.master = master
        self.train_color_dict = get_index(TRAIN_COLOR_INDEX_PATH)
        self.train_sift_dict = get_index(TRAIN_SIFT_INDEX_PATH)
        self.sift_descriptor = SIFTDescriptor(VISUAL_VOCABULARY_PATH)

        topframe = Frame(self.master)
        topframe.pack()

        #Buttons
        topspace = Label(topframe).grid(row=0, columnspan=2)
        self.bbutton= Button(topframe, text=" Choose an image ", command=self.browse_query_img)
        self.bbutton.grid(row=1, column=1)
        self.cbutton = Button(topframe, text=" Search ", command=self.show_results_imgs)
        self.cbutton.grid(row=1, column=2)
        downspace = Label(topframe).grid(row=3, columnspan=4)

        self.master.mainloop()


    def browse_query_img(self):
        try:
            self.query_img_frame.destroy()
            self.result_img_frame.destroy()
        except AttributeError:
            None

        self.query_img_frame = Frame(self.master)
        self.query_img_frame.pack()
        from tkFileDialog import askopenfilename
        self.filename = tkFileDialog.askopenfile(title='Choose an Image File').name

        # show query image
        image_file = Image.open(self.filename)
        resized = image_file.resize((100, 100), Image.ANTIALIAS)
        im = ImageTk.PhotoImage(resized)
        image_label = Label(self.query_img_frame, image=im)
        image_label.pack()

        self.query_img_frame.mainloop()

    def show_results_imgs(self):
        self.result_img_frame = Frame(self.master)
        self.result_img_frame.pack()

        # perform the search
        results = self.retrieve_images(self.filename)

        # show result pictures
        COLUMNS = 5
        image_count = 0
        for (score, resultID) in results:
            # load the result image and display it
            image_count += 1
            r, c = divmod(image_count - 1, COLUMNS)
            im = Image.open( self.search_path + "/" + resultID)
            resized = im.resize((100, 100), Image.ANTIALIAS)
            tkimage = ImageTk.PhotoImage(resized)
            myvar = Label(self.result_img_frame, image=tkimage)
            myvar.image = tkimage
            myvar.grid(row=r, column=c)

        self.result_img_frame.mainloop()

    def calculate_score(self, similarities):
        weights = np.array([1, 2])
        return np.inner(weights, similarities)

    def retrieve_images(self, query_path):
        query_image = cv2.imread(query_path)
        results = []

        # Compute color feature of query image
        query_color = describe_color(query_image)

        # Compute sift feature
        query_sift = self.sift_descriptor.describe(cv2.cvtColor(query_image, 0))

        # Compute deep learning predictions of query image
        # predictions = inference(query_path)

        for file_name, train_value in self.train_color_dict.iteritems():
            train_label = train_value[0]
            train_color = train_value[1]

            # Compute similarity of different features
            color_sim = color_similarity(query_color, train_color)
            sift_sim = color_similarity(query_sift, self.train_sift_dict.get(file_name)[1])
            # learning_sim = learning_similarity(predictions, train_label)

            score = self.calculate_score([color_sim, sift_sim])
            results.append((score, file_name))

        top_k = sorted(results, key=lambda x: x[0], reverse=True)[:K_SIZE]

        return top_k

root = Tk()
window = UI_class(root,'../data/train/')