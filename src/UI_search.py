from Tkinter import *
from PIL import Image, ImageTk
from searcher import Searcher
import time
import tkFileDialog
import numpy as np

DISPLAY_IMAGE_COLUMNS = 4


def get_weights(color_check=1, word_check=1, learning_check=1, tag_check=1):
    if sum([color_check, word_check, learning_check, tag_check]) == 1:
        return np.array([color_check, word_check, learning_check, tag_check])
    elif sum([color_check, word_check, learning_check, tag_check]) == 4:
        return np.array([7.87123598, 7.29097985, 33.38541097, 1])

    return np.array([7.87123598, 7.29097985, 33.38541097, 1])


class GUI:
    def __init__(self, master, search_path):
        self.search_path = search_path
        self.master = master
        self.searcher = Searcher()
        self.default_weights = self.searcher.get_weights()

        self.file_name = None
        self.result_image_frame = Frame(self.master)
        self.query_image_frame = Frame(self.master)

        top_frame = Frame(self.master)
        top_frame.pack()

        # Buttons
        topspace = Label(top_frame).grid(row=0, columnspan=2)
        self.browse_button = Button(top_frame, text=" Choose an image ", command=self.browse_query_image)
        self.browse_button.grid(row=1, column=1)
        self.search_button = Button(top_frame, text=" Search ", command=self.show_results_images, width=20, bd=3)
        self.search_button.grid(row=1, column=3)

        self.color_check = IntVar()
        self.word_check = IntVar()
        self.learning_check = IntVar()

        # check everything
        self.color_check.set(1)
        self.word_check.set(1)
        self.learning_check.set(1)

        color_checkbox = Checkbutton(top_frame, text="Color Histogram", variable=self.color_check, onvalue=1, offvalue=0, height=5)
        word_checkbox = Checkbutton(top_frame, text="Visual Words", variable=self.word_check, onvalue=1, offvalue=0, height=5)
        learning_checkbox = Checkbutton(top_frame, text="Deep Learning", variable=self.learning_check, onvalue=1, offvalue=0, height=5)

        color_checkbox.grid(row=2, column=1)
        word_checkbox.grid(row=2, column=2)
        learning_checkbox.grid(row=2, column=3)

        tag_label = Label(top_frame, text="Tags ")
        tag_label.grid(row=3, column=1)
        self.tag_box = Entry(top_frame)
        self.tag_box.grid(row=3, column=2)
        downspace = Label(top_frame).grid(row=4, columnspan=4)

        self.master.mainloop()

    def browse_query_image(self):
        try:
            self.query_image_frame.destroy()
            self.result_image_frame.destroy()
        except AttributeError:
            None

        self.query_image_frame = Frame(self.master)
        self.query_image_frame.pack()
        self.file_name = tkFileDialog.askopenfile(title='Choose an Image File').name

        # show query image
        image_file = Image.open(self.file_name)
        resized = image_file.resize((100, 100), Image.ANTIALIAS)
        im = ImageTk.PhotoImage(resized)
        image_label = Label(self.query_image_frame, image=im)
        image_label.pack()

        self.query_image_frame.mainloop()

    def show_results_images(self):
        try:
            self.result_image_frame.destroy()
        except AttributeError:
            None

        tags = self.tag_box.get()
        if tags == "":
            tags = []
            tag_check = 0
        else:
            tags = tags.split()
            tag_check = 1

        weights = get_weights(self.color_check.get(), self.word_check.get(), self.learning_check.get(), tag_check)
        self.searcher.set_weights(weights)
        text_only = self.color_check.get() + self.word_check.get() + self.learning_check.get() == 0


        self.result_image_frame = Frame(self.master)
        self.result_image_frame.pack()

        # perform the search
        start = time.time()
        if text_only or self.file_name is None:
            try:
                self.query_image_frame.destroy()
            except AttributeError:
                None
            results = self.searcher.retrieve_by_tags(tags)
        else:
            results = self.searcher.retrieve_images(self.file_name, tags)
        end = time.time()
        print "Search time: %s" % str(end - start)

        # show result pictures
        image_count = 0
        for (score, image_id) in results:
            # load the result image and display it
            image_count += 1
            row, column = divmod(image_count - 1, DISPLAY_IMAGE_COLUMNS)
            im = Image.open(self.search_path + "/" + image_id)
            resized_image = im.resize((100, 100), Image.ANTIALIAS)
            tkimage = ImageTk.PhotoImage(resized_image)
            myvar = Label(self.result_image_frame, image=tkimage)
            myvar.image = tkimage
            myvar.grid(row=row, column=column)

        self.result_image_frame.mainloop()


root = Tk()
root.title("Image Retrieval")
window = GUI(root, '../data/train/')
