"""
Object detection algorithm.
"""
import re

import numpy as np
import cv2

import torch
from torch.backends import cudnn

import os
import signal

import tkinter as tk
from PIL import Image, ImageTk

from myutils import *


def tkinter_design():
    window = tk.Tk()
    window.wm_title("Object Detector")

    frame = tk.Frame(master=window, relief=tk.FLAT, height=250, width=250)
    frame.grid(row=0, column=0, padx=5, pady=5)

    window.columnconfigure(0, weight=1, minsize=50)
    window.rowconfigure(0, weight=1, minsize=50)

    # # Capture video frames
    # cap = cv2.VideoCapture(0)

    def callback_start():
        # open_video()
        button.destroy()
        w.destroy()
        mod_menu.destroy()

        model = model_selection(chosen_model, ALL_MODELS)

        imageFrame = tk.Frame(master=window, width=1280, height=720)
        imageFrame.grid(row=0, column=0, padx=10, pady=2)
        lmain = tk.Label(imageFrame)
        lmain.grid(row=0, column=0)

        if chosen_model.get() == "Yolov5":
            show_frame_pytorch(lmain, trigger, model)
        else:
            show_frame_tensorflow(lmain, trigger, model, cat_index, COCO17_HUMAN_POSE_KEYPOINTS)


    button = tk.Button(text="Start Video", width=12, height=8,
                       bg="green", fg="black", master=frame,
                       command=callback_start)
    button.grid(row=0, column=0, sticky="nsew", padx=5, pady=5)

    trigger = tk.StringVar(frame)
    trigger.set("flip frame horizontally")  # default value

    button_frame = tk.Frame(master=frame, relief=tk.FLAT)
    button_frame.grid(row=0, column=1, padx=5, pady=5)

    w = tk.OptionMenu(button_frame, trigger, "flip frame horizontally", "don't flip frame")
    w.grid(row=0, column=0, padx=5, pady=5)


    ALL_MODELS, COCO17_HUMAN_POSE_KEYPOINTS, _, cat_index = initialize()
    models_list = set(ALL_MODELS.keys())

    # Choose only EfficientDet models
    r = re.compile(r'EfficientDet([\D]+\d)')
    models_list = sorted([*filter(r.match, models_list)])
    # models_list = sorted([r.match(f).group() for f in models_list if r.search(f) != None])
    models_list.insert(0, "Yolov5")

    chosen_model = tk.StringVar(frame)
    chosen_model.set("Yolov5")

    mod_menu = tk.OptionMenu(button_frame, chosen_model, *models_list)
    mod_menu.grid(row=1, column=0, padx=5, pady=5)

    window.mainloop()


def show_frame_pytorch(lmain, trigger, model):
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
    lmain.after(20, lambda: show_frame_pytorch(lmain, trigger, model))


def show_frame_tensorflow(lmain, trigger, model, category_index, COCO17_HUMAN_POSE_KEYPOINTS):
    _, frame = cap.read()
    if trigger.get() == "flip frame horizontally":
        frame = cv2.flip(frame, 1)

    with tf.device("/GPU:0"):
        results = model(np.expand_dims(frame, 0))
    result = {key:value.numpy() for key,value in results.items()}
    label_id_offset = 0
    keypoints, keypoint_scores = None, None
    if 'detection_keypoints' in result:
        keypoints = result['detection_keypoints'][0]
        keypoint_scores = result['detection_keypoint_scores'][0]
    viz_utils.visualize_boxes_and_labels_on_image_array(
        frame,
        result['detection_boxes'][0],
        (result['detection_classes'][0] + label_id_offset).astype(int),
        result['detection_scores'][0],
        category_index,
        use_normalized_coordinates=True,
        max_boxes_to_draw=200,
        min_score_thresh=.30,
        agnostic_mode=False,
        keypoints=keypoints,
        keypoint_scores=keypoint_scores,
        keypoint_edges=COCO17_HUMAN_POSE_KEYPOINTS)

    cv2image = cv2.cvtColor(frame, cv2.COLOR_BGR2RGBA)
    img = Image.fromarray(cv2image)
    img = img.resize((1280, 960), Image.ANTIALIAS)

    imgtk = ImageTk.PhotoImage(image=img)
    lmain.imgtk = imgtk
    lmain.configure(image=imgtk)
    lmain.after(20, lambda: show_frame_tensorflow(lmain, trigger, model, category_index, COCO17_HUMAN_POSE_KEYPOINTS))


def model_selection(chosen_model, ALL_MODELS):
    if chosen_model.get() == "Yolov5":
        device = "cuda" if torch.cuda.is_available() else "cpu"

        # Model originally on GPU? If not cast to cpu it doesn't open display.

        if device == "cuda":
            model = torch.hub.load("ultralytics/yolov5", "yolov5l6", pretrained=True)
            model.half()
        else:
            model = torch.hub.load("ultralytics/yolov5", "yolov5s6", pretrained=True)
            model.to(device)

    else:
        with tf.device("/GPU:0"):
            model = hub.load(ALL_MODELS[chosen_model.get()])

    return model


if __name__ == '__main__':
    cudnn.benchmark = True

    gpu = tf.config.list_physical_devices()[1]
    tf.config.experimental.set_memory_growth(gpu, True)

    # Capture video frames
    cap = cv2.VideoCapture(0)

    tkinter_design()

    os.kill(os.getpid(), signal.SIGTERM)