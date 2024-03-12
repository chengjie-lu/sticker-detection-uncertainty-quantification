from PIL import ImageTk, Image

from tkinter import messagebox
from tkinter import ttk
import cv2

import tkinter as tk
import tkinter.font as tkFont

import time

import cv2 as cv
import torch

from utils import init_camera, grab_one_image, load_camera_calibration, load_model, run_model, calc_3d_point

IMAGE_FRAME_SIZE = 0.85
SLIDER_FRAME_SIZE = 0.05
BUTTONS_FRAME_SIZE = 0.1

NUM_WORKERS = 1
BATCH_SIZE = 6

NUM_CLASSES = 3  # logo + sticker + background

WEIGHT_DECAY = 0.0005
MOMENTUM = 0.9
LEARNING_RATE = 0.005

CONFIG = {
    "lr": LEARNING_RATE,
    "momentum": MOMENTUM,
    "weight_decay": WEIGHT_DECAY,
    "batch_size": BATCH_SIZE
}

MODEL_NAME = 'retinanet_resnet50_fpn_v2'

CHECKPOINT_PATH = 'checkpoints/retinanet_resnet50_fpn_v2_aug/version_0/checkpoints/epoch=31-step=7712.ckpt'

RUNTIME_TYPE = 'normal'  # Choices == 'onnx' and 'normal'

LABELS = {'Background': 0, 'Logo': 1, 'Sticker': 2}


class App:
    def __init__(self, root):
        # setting title
        self.root = root
        self.root.title("Sticker Detector")
        # setting window size
        width = 600
        height = 800
        screenwidth = root.winfo_screenwidth()
        screenheight = root.winfo_screenheight()
        alignstr = '%dx%d+%d+%d' % (width, height, (screenwidth - width) / 2, (screenheight - height) / 2)
        root.geometry(alignstr)
        root.resizable(width=True, height=True)

        # make a frame for a slider above the video using place
        self.slider_frame = tk.Frame(root)
        self.slider_frame.place(relx=0.0, rely=0.00, relwidth=1.0, relheight=SLIDER_FRAME_SIZE, anchor='nw')

        # make a large frame for the video using place
        self.vid_frame = tk.Frame(root, bg='black')
        self.vid_frame.place(relx=0.0, rely=SLIDER_FRAME_SIZE, relwidth=1.0, relheight=IMAGE_FRAME_SIZE, anchor='nw')

        self.button_frame = tk.Frame(root)
        self.button_frame.place(relx=0.0, rely=SLIDER_FRAME_SIZE + IMAGE_FRAME_SIZE, relwidth=1.0,
                                relheight=BUTTONS_FRAME_SIZE, anchor='nw')

        self.image_label = tk.Label(self.vid_frame)
        self.image_label.pack()

        self.root.bind('<Key>', self.my_callback)  # Any Key  is pressed

        # make a slider in slider_frame
        self.image_resize_variable = tk.IntVar()
        self.image_resize_variable.set(100)
        self.slider = tk.Scale(self.slider_frame, from_=1, to=100, orient='horizontal',
                               variable=self.image_resize_variable)
        self.slider.pack(side='left', fill='both', expand=True)

        # make 4 buttons under the video named "Sticker", "Logo", "Increase", "Decrease".
        # It should only fill about 1/4 of the screen
        self.sticker_button = tk.Button(self.button_frame, text="Sticker", command=self.sticker_button_cmd)
        self.sticker_button.pack(side='left', fill='both', expand=True)
        self.sticker_button.configure(bg='green')

        self.logo_button = tk.Button(self.button_frame, text="Logo", command=self.logo_button_cmd)
        self.logo_button.pack(side='left', fill='both', expand=True)
        self.logo_button.configure(bg='green')

        self.increase_button = tk.Button(self.button_frame, text="Increase", command=self.increase_button_cmd)
        self.increase_button.pack(side='left', fill='both', expand=True)

        self.decrease_button = tk.Button(self.button_frame, text="Decrease", command=self.decrease_button_cmd)
        self.decrease_button.pack(side='left', fill='both', expand=True)

        self.min_score = 0.6
        # self.camera = init_camera()
        self.p, self.d, self.dist_maps = load_camera_calibration()
        # load model
        self.model = load_model(RUNTIME_TYPE)

        self.show_bounding_box_sticker = True
        self.show_bounding_box_logo = True

        self.display_image()

    def display_image(self):

        with torch.no_grad():
            start = time.time()
            # image_og = grab_one_image(self.camera)
            image_og = cv2.imread("test_images/test.jpg")
            # Undistorts images with the maps calculated in load_camera_calibration()
            image_og = cv.remap(image_og, self.dist_maps[0], self.dist_maps[1], cv.INTER_LINEAR)
            # image_og = cv.undistort(image_og, self.p, self.d)
            scale = self.image_resize_variable.get() / 100.0

            # print('scale', scale)
            # image_og = cv.rotate(image_og, cv.ROTATE_90_CLOCKWISE)

            image_rz = cv.resize(image_og, (0, 0), fx=scale, fy=scale)
            # flip image 90 degrees
            # image_rz = cv.rotate(image_rz, cv.ROTATE_90_CLOCKWISE)

            preds = run_model(image_rz, self.model, RUNTIME_TYPE)

            for j in range(len(preds[0]['scores']) - 1, -1, -1):
                if preds[0]['scores'][j] < self.min_score or preds[0]['labels'][j] == 0 or (
                        preds[0]['labels'][j] == 1 and not self.show_bounding_box_logo):
                    preds[0]['boxes'] = torch.cat((preds[0]['boxes'][:j], preds[0]['boxes'][j + 1:]))
                    preds[0]['labels'] = torch.cat((preds[0]['labels'][:j], preds[0]['labels'][j + 1:]))
                    preds[0]['scores'] = torch.cat((preds[0]['scores'][:j], preds[0]['scores'][j + 1:]))

            for pred in preds:
                for key, value in pred.items():
                    pred[key] = value.cpu()

            boxes = preds[0]['boxes'].cpu().detach().numpy()
            labels = preds[0]['labels'].cpu().detach().numpy()
            scores = preds[0]['scores'].cpu().detach().numpy()

            # multiply all values in boxes by 5 to get back to original size
            boxes = boxes / scale

            labels = [key for value in labels for key, val in LABELS.items() if val == value]
            for box, label, score in zip(boxes, labels, scores):
                if self.show_bounding_box_sticker:
                    if label == 'Sticker':
                        image_og = cv.rectangle(image_og, (int(box[0]), int(box[1])), (int(box[2]), int(box[3])),
                                                (255, 0, 0), 2)
                        image_og = cv.putText(image_og, str(label + ' ' + str(score)[:5]), (int(box[0]), int(box[1])),
                                              cv.FONT_HERSHEY_SIMPLEX, 2, (255, 0, 0), 3,
                                              cv.LINE_AA)  # (255,0,0) is blue

                        x = int((box[0] + box[2]) / 2)
                        y = int((box[1] + box[3]) / 2)
                        center_3d_str = calc_3d_point(box, self.p)
                        # draw
                        image_og = cv.circle(image_og, (int(x), int(y)), 3, (0, 0, 255), -1)
                        image_og = cv.putText(image_og, center_3d_str, (int(x), int(y)), cv.FONT_HERSHEY_SIMPLEX, 2,
                                              (0, 0, 255), 2, cv.LINE_AA)

                if self.show_bounding_box_logo:
                    if label == 'Logo':
                        image_og = cv.rectangle(image_og, (int(box[0]), int(box[1])), (int(box[2]), int(box[3])),
                                                (255, 0, 0), 2)
                        image_og = cv.putText(image_og, str(label + ' ' + str(score)[:5]), (int(box[0]), int(box[1])),
                                              cv.FONT_HERSHEY_SIMPLEX, 2, (255, 0, 0), 3,
                                              cv.LINE_AA)  # (255,0,0) is blue

                        x = int((box[0] + box[2]) / 2)
                        y = int((box[1] + box[3]) / 2)
                        center_3d_str = calc_3d_point(box, self.p)
                        # draw
                        image_og = cv.circle(image_og, (int(x), int(y)), 3, (0, 0, 255), -1)
                        image_og = cv.putText(image_og, center_3d_str, (int(x), int(y)), cv.FONT_HERSHEY_SIMPLEX, 2,
                                              (0, 0, 255), 2, cv.LINE_AA)

            # end time
            fps = 1 / (time.time() - start)
            # write fps in top right corner of image_og in size 2. Only show 4 digits
            cv.putText(image_og, 'FPS: ' + str('%.2f' % fps), (10, 60), cv.FONT_HERSHEY_SIMPLEX, 2, (255, 0, 0), 3,
                       cv.LINE_AA)
            # write min score underneath
            cv.putText(image_og, 'Threshhold Score: ' + str('%.2f' % self.min_score), (10, 130),
                       cv.FONT_HERSHEY_SIMPLEX, 2, (255, 0, 0), 3, cv.LINE_AA)

            self.root.update_idletasks()
            vid_frame_width = self.root.winfo_width()
            vid_frame_height = int(self.root.winfo_height() * IMAGE_FRAME_SIZE)
            # Open the image file
            # Resize the image to fit in the frame
            img = cv2.resize(image_og, (vid_frame_width, vid_frame_height))
            # Convert the image to a format that can be used in a tkinter label
            img = cv2.cvtColor(img, cv2.COLOR_BGR2RGB)
            img = Image.fromarray(img)

            img = ImageTk.PhotoImage(img)
            # Create a label with the image

            self.image_label.configure(image=img)
            self.image_label.image = img  # Keep a reference to the image to prevent it from being garbage collected

            # Call this method again after 1000ms (1 second)
            self.vid_frame.after(10, self.display_image)

    def logo_button_cmd(self):

        self.show_bounding_box_logo = not self.show_bounding_box_logo
        if self.show_bounding_box_logo:
            self.logo_button.configure(bg='green')
            print("Now showing logo")
        else:
            self.logo_button.configure(bg='red')
            print("Not showing logo")

    def sticker_button_cmd(self):

        self.show_bounding_box_sticker = not self.show_bounding_box_sticker
        if self.show_bounding_box_sticker:
            self.sticker_button.configure(bg='green')
            print("Now showing sticker")
        else:
            self.sticker_button.configure(bg='red')
            print("Not showing sticker")

    def increase_button_cmd(self):
        self.min_score += 0.01

    def decrease_button_cmd(self):
        self.min_score -= 0.01

    def my_callback(self, event):  # When any key is pressed
        if event.char == 's':
            self.show_bounding_box_sticker = not self.show_bounding_box_sticker
            if self.show_bounding_box_sticker:
                self.sticker_button.configure(bg='green')
                print("Now showing sticker")
            else:
                self.sticker_button.configure(bg='red')
                print("Not showing sticker")
        elif event.char == 'l':
            self.show_bounding_box_logo = not self.show_bounding_box_logo
            if self.show_bounding_box_logo:
                self.logo_button.configure(bg='green')
                print("Now showing logo")
            else:
                self.logo_button.configure(bg='red')
                print("Not showing logo")
        # if arrow up is pressed
        elif event.keycode == 38:
            self.min_score += 0.01
            print("min score: ", self.min_score)
        # if arrow down is pressed
        elif event.keycode == 40:
            self.min_score -= 0.01
            print("min score: ", self.min_score)
        # if escape is pressed
        elif event.keycode == 27:
            self.root.destroy()
            print("exiting")
            exit()
        else:
            print(f"key pressed: {event.char}, which does nothing")


if __name__ == "__main__":
    roo = tk.Tk()
    app = App(roo)
    roo.mainloop()
