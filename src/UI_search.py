from Tkinter import *
import tkFileDialog
from PIL import Image, ImageTk
from searcher import Searcher
import time


class GUI:
    def __init__(self, master, search_path):
        self.search_path = search_path
        self.master = master
        self.searcher = Searcher()

        topframe = Frame(self.master)
        topframe.pack()

        # Buttons
        topspace = Label(topframe).grid(row=0, columnspan=2)
        self.bbutton = Button(topframe, text=" Choose an image ", command=self.browse_query_img)
        self.bbutton.grid(row=1, column=1)
        self.cbutton = Button(topframe, text=" Search ", command=self.show_results_imgs, width=20, bd=3)
        self.cbutton.grid(row=1, column=3)

        self.ch = IntVar()
        self.vw = IntVar()
        self.dl = IntVar()

        # check everything
        self.ch.set(1)
        self.vw.set(1)
        self.dl.set(1)

        ch_checkbox = Checkbutton(topframe, text="CH", variable=self.ch, onvalue=1, offvalue=0, height=5)
        vw_checkbox = Checkbutton(topframe, text="VW", variable=self.vw, onvalue=1, offvalue=0, height=5)
        dl_checkbox = Checkbutton(topframe, text="DL", variable=self.dl, onvalue=1, offvalue=0, height=5)

        ch_checkbox.grid(row=2, column=1)
        vw_checkbox.grid(row=2, column=2)
        dl_checkbox.grid(row=2, column=3)

        taglabel = Label(topframe, text="Tags ")
        taglabel.grid(row=3, column=1)
        self.tagbox = Entry(topframe)
        self.tagbox.grid(row=3, column=2)
        downspace = Label(topframe).grid(row=4, columnspan=4)

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
        try:
            self.result_img_frame.destroy()
        except AttributeError:
            None

        self.searcher.weights *= [self.ch.get(), self.vw.get(), self.dl.get()]
        tags = self.tagbox.get()
        self.result_img_frame = Frame(self.master)
        self.result_img_frame.pack()

        # perform the search
        start = time.time()
        results = self.searcher.retrieve_images(self.filename)
        end = time.time()
        print "Search time: %s" % str(end - start)

        # show result pictures
        COLUMNS = 5
        image_count = 0
        for (score, resultID) in results:
            # load the result image and display it
            image_count += 1
            r, c = divmod(image_count - 1, COLUMNS)
            im = Image.open(self.search_path + "/" + resultID)
            resized = im.resize((100, 100), Image.ANTIALIAS)
            tkimage = ImageTk.PhotoImage(resized)
            myvar = Label(self.result_img_frame, image=tkimage)
            myvar.image = tkimage
            myvar.grid(row=r, column=c)

        self.result_img_frame.mainloop()


root = Tk()
window = GUI(root, '../data/train/')
