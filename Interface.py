import tkinter
import PIL.Image, PIL.ImageTk
import time
import os
from tkinter import filedialog
from Predict import *


class App:
    def __init__(self, window, window_title, video_source=0):
        self.window = window
        #self.window.iconphoto(True, PIL.ImageTk.PhotoImage(file="../Resources/Doofenshmirtz_Portrait.png"))
        self.window.geometry("1400x900+50+50")
        self.window.resizable(width=True, height=True)
        self.window.title(window_title)
        self.video_source = video_source
        self.delay = 15
        self.running = ""
        self.dirname = "Snapshots"

        self.listbox = tkinter.Listbox(self.window)
        self.pack_and_place(self.listbox, height=300, width=200, x=15, y=50)
        self.filelist = []
        self.get_filelist()

        self.load_img_list_btn = tkinter.Button(self.window, text='Load Selected Image', command=self.load_selected_image)
        self.pack_and_place(self.load_img_list_btn, height=25, width=150, x=15, y=375)

        self.load_directory_list_btn = tkinter.Button(self.window, text='Change dir',   command=self.load_directory)
        self.pack_and_place(self.load_directory_list_btn, height=25, width=150, x=15, y=15)

        self.canvas = tkinter.Canvas(self.window, width=1024, height=768)
        self.pack_and_place(self.canvas, x=250, y=50)

        self.load_img_btn = tkinter.Button(self.window, text='Load Image', command=self.open_img)
        self.pack_and_place(self.load_img_btn, height=25, width=100, x=632, y=15)

        self.load_video_btn = tkinter.Button(self.window, text='Capture Video', command=self.capture_video)
        self.pack_and_place(self.load_video_btn, height=25, width=100, x=800, y=15)

        self.what_do_i_see_btn = tkinter.Button(self.window, text='What do I see?', command=run_talk)
        self.hide_btn(self.what_do_i_see_btn)

        # open video source (by default this will try to open the computer webcam)
        self.vid = MyVideoCapture(self.video_source)
        self.img_obj = MyLoadImg()

        # Button that lets the user take a snapshot
        self.btn_snapshot = tkinter.Button(window, text="Snapshot", width=50, command=self.snapshot)
        self.hide_btn(self.btn_snapshot)

        self.window.mainloop()

    def snapshot(self):
        # Get a frame from the video source
        ret, frame = self.vid.get_frame()

        if ret:
            cv2.imwrite("Snapshots/frame-" + time.strftime("%d-%m-%Y-%H-%M-%S") + ".jpg",
                        cv2.cvtColor(frame, cv2.COLOR_RGB2BGR))
            self.get_filelist()

    def update(self):
        # Get a frame from the video source
        ret, frame = self.vid.get_frame()
        if ret:
            self.photo = PIL.ImageTk.PhotoImage(image=PIL.Image.fromarray(frame))
            self.canvas.create_image(0, 0, image=self.photo, anchor=tkinter.NW)

        if self.running == "Video":
            self.window.after(self.delay, self.update)

    def load_selected_image(self):
        self.running = "Img"
        x = self.listbox.curselection()[0]
        self.open_img(f"{self.dirname}/{self.listbox.get(x)}")

    def open_img(self, filename=""):
        self.running = "Img"
        ret, photo = self.img_obj.get_img(filename)

        if ret:
            self.hide_btn(self.btn_snapshot)
            self.hide_btn(self.what_do_i_see_btn)
            self.canvas.delete("all")
            photo = cv2.resize(photo, (1024, 768))

            self.photo = PIL.ImageTk.PhotoImage(image=PIL.Image.fromarray(photo))  # ???
            self.canvas.create_image(0, 0, image=self.photo, anchor=tkinter.NW)

            run_talk()
        return

    def capture_video(self):
        self.running = "Video"
        self.show_btn(self.btn_snapshot, height=25, width=100, x=712, y=830)
        self.show_btn(self.what_do_i_see_btn, height=25, width=100, x=600, y=830)
        self.canvas.delete("all")

        # After it is called once, the update method will be automatically called every delay milliseconds
        self.update()
        return

    def get_filelist(self):
        self.listbox.delete(0, tkinter.END)
        self.filelist = os.listdir(self.dirname)
        for file in self.filelist:
            self.listbox.insert(tkinter.END, file)

    def hide_btn(self, btn):
        btn.pack_forget()
        btn.place_forget()

    def show_btn(self, btn, height, width, x, y):
        btn.pack(anchor=tkinter.CENTER, expand=True)
        btn.place(bordermode=tkinter.OUTSIDE, height=height, width=width, x=x, y=y)

    def load_directory(self):
        self.dirname = filedialog.askdirectory()
        self.get_filelist()

    def pack_and_place(self, obj, x, y, height=-1, width=-1):
        obj.pack()
        if height == -1 and width == -1:
            obj.place(bordermode=tkinter.OUTSIDE, x=x, y=y)
        else:
            obj.place(bordermode=tkinter.OUTSIDE, height=height, width=width, x=x, y=y)


class MyVideoCapture:
    def __init__(self, video_source=0):
        # Open the video source
        self.vid = cv2.VideoCapture(video_source, cv2.CAP_DSHOW)
        if not self.vid.isOpened():
            raise ValueError("Unable to open video source", video_source)

        # Get video source width and height
        self.width = self.vid.get(cv2.CAP_PROP_FRAME_WIDTH)
        self.height = self.vid.get(cv2.CAP_PROP_FRAME_HEIGHT)

    def get_frame(self):
        if self.vid.isOpened():
            ret, frame = self.vid.read()
            boxed_image = object_detection_api(frame, "", threshold=0.8, rect_th=1, text_size=1, text_th=1)
            resized_cv_frame = cv2.resize(boxed_image, (1024, 768))

            if ret:
                # Return a boolean success flag and the current frame converted to BGR
                return ret, resized_cv_frame
            else:
                return ret, None
        else:
            return False, None

    # Release the video source when the object is destroyed
    def __del__(self):
        if self.vid.isOpened():
            self.vid.release()


class MyLoadImg:
    def __init__(self):
        self.filename = ""

    def get_img(self, filename):
        if filename == "":
            self.filename = filedialog.askopenfilename(title='open')
            if self.filename == "":
                return False, None, None
        else:
            self.filename = filename
        boxed_image = object_detection_api(None, self.filename, threshold=0.8)
        return True, boxed_image


config_model()
init_tts()

# Create a window and pass it to the Application object
MyApp = App(tkinter.Tk(), "BlindVision")
cv2.destroyAllWindows()

