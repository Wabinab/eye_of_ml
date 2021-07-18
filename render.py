"""
Object detection algorithm.
"""
import numpy as np
import cv2

import torch

import os
import signal

import tkinter as tk
from PIL import Image, ImageTk


def tkinter_design():
    window = tk.Tk()
    window.wm_title("Object Detector")

    frame = tk.Frame(master=window, relief=tk.FLAT, height=250, width=250)
    frame.grid(row=0, column=0, padx=5, pady=5)

    window.columnconfigure(0, weight=1, minsize=50)
    window.rowconfigure(0, weight=1, minsize=50)

    # Capture video frames
    cap = cv2.VideoCapture(0)

    def callback_start():
        # open_video()
        button.destroy()
        w.destroy()

        imageFrame = tk.Frame(master=window, width=1280, height=720)
        imageFrame.grid(row=0, column=0, padx=10, pady=2)
        lmain = tk.Label(imageFrame)
        lmain.grid(row=0, column=0)

        show_frame(lmain)


    def show_frame(lmain):
        _, frame = cap.read()
        if trigger.get() == "flip frame horizontally":
            frame = cv2.flip(frame, 1)
        results = model(frame)
        results.display(render=True)
        cv2image = cv2.cvtColor(frame, cv2.COLOR_BGR2RGBA)
        img = Image.fromarray(cv2image)
        img = img.resize((1280, 960), Image.ANTIALIAS)
        imgtk = ImageTk.PhotoImage(image=img)
        lmain.imgtk = imgtk
        lmain.configure(image=imgtk)
        lmain.after(20, lambda: show_frame(lmain))


    button = tk.Button(text="Start Video", width=12, height=8,
                       bg="green", fg="black", master=frame,
                       command=callback_start)
    button.grid(row=0, column=0, sticky="nsew", padx=5, pady=5)

    trigger = tk.StringVar(frame)
    trigger.set("flip frame horizontally")  # default value

    w = tk.OptionMenu(frame, trigger, "flip frame horizontally", "don't flip frame")
    w.grid(row=0, column=1, padx=5, pady=5)

    window.mainloop()


# def open_video():
#     cv2.namedWindow("preview")
#     vc = cv2.VideoCapture(0)
#
#     if vc.isOpened():
#         rval, frame = vc.read()
#     else:
#         rval = False
#
#     while rval:
#         results = model(frame)
#         results.display(render=True)
#         cv2.imshow("preview", frame)
#         rval, frame = vc.read()
#
#         key = cv2.waitKey(20)
#         if key == 27:  # ESC key.
#             break
#
#     vc.release()
#     cv2.destroyWindow("preview")


if __name__ == '__main__':
    model = torch.hub.load("ultralytics/yolov5", "yolov5l6", pretrained=True)
    model.half()

    tkinter_design()

    os.kill(os.getpid(), signal.SIGTERM)