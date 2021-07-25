"""
Object detection algorithm.
"""
import numpy as np
import cv2
import re

import torch
from torch.backends import cudnn

from efficientdet.utils import BBoxTransform, ClipBoxes
from backbone import EfficientDetBackbone
from myutils.myutils import invert_affine, postprocess, preprocess_video

import os
import signal

import tkinter as tk
from PIL import Image, ImageTk

use_float16 = False


obj_list = ['person', 'bicycle', 'car', 'motorcycle', 'airplane', 'bus', 'train', 'truck', 'boat', 'traffic light',
            'fire hydrant', '', 'stop sign', 'parking meter', 'bench', 'bird', 'cat', 'dog', 'horse', 'sheep',
            'cow', 'elephant', 'bear', 'zebra', 'giraffe', '', 'backpack', 'umbrella', '', '', 'handbag', 'tie',
            'suitcase', 'frisbee', 'skis', 'snowboard', 'sports ball', 'kite', 'baseball bat', 'baseball glove',
            'skateboard', 'surfboard', 'tennis racket', 'bottle', '', 'wine glass', 'cup', 'fork', 'knife', 'spoon',
            'bowl', 'banana', 'apple', 'sandwich', 'orange', 'broccoli', 'carrot', 'hot dog', 'pizza', 'donut',
            'cake', 'chair', 'couch', 'potted plant', 'bed', '', 'dining table', '', '', 'toilet', '', 'tv',
            'laptop', 'mouse', 'remote', 'keyboard', 'cell phone', 'microwave', 'oven', 'toaster', 'sink',
            'refrigerator', '', 'book', 'clock', 'vase', 'scissors', 'teddy bear', 'hair drier',
            'toothbrush']


class Colors:
    # Ultralytics color palette https://ultralytics.com/
    def __init__(self):
        # hex = matplotlib.colors.TABLEAU_COLORS.values()
        hex = ('FF3838', 'FF9D97', 'FF701F', 'FFB21D', 'CFD231', '48F90A', '92CC17', '3DDB86', '1A9334', '00D4BB',
               '2C99A8', '00C2FF', '344593', '6473FF', '0018EC', '8438FF', '520085', 'CB38FF', 'FF95C8', 'FF37C7')
        self.palette = [self.hex2rgb('#' + c) for c in hex]
        self.n = len(self.palette)

    def __call__(self, i, bgr=False):
        c = self.palette[int(i) % self.n]
        return (c[2], c[1], c[0]) if bgr else c

    @staticmethod
    def hex2rgb(h):  # rgb order (PIL)
        return tuple(int(h[1 + i:1 + i + 2], 16) for i in (0, 2, 4))


colors = Colors()  # create instance for 'from utils.plots import colors'


def display(preds, imgs):
    for i in range(len(imgs)):
        if len(preds[i]['rois']) == 0:
            return imgs[i]

        for j in range(len(preds[i]['rois'])):
            (x1, y1, x2, y2) = preds[i]['rois'][j].astype(np.int)
            cv2.rectangle(imgs[i], (x1, y1), (x2, y2), colors(j), 2)
            obj = obj_list[preds[i]['class_ids'][j]]
            score = float(preds[i]['scores'][j])

            cv2.putText(imgs[i], f"{obj}: {score:.3f}",
                        (x1 + 3, y1 - 10), cv2.FONT_HERSHEY_SIMPLEX, 0.5,
                        colors(j), 1, lineType=cv2.LINE_AA)

        return imgs[i]


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

        imageFrame = tk.Frame(master=window, width=1280, height=720)
        imageFrame.grid(row=0, column=0, padx=10, pady=2)
        lmain = tk.Label(imageFrame)
        lmain.grid(row=0, column=0)

        model = get_model(chosen_model.get())

        if chosen_model.get() == "Yolov5":
            show_frame(lmain, trigger, model)
        else:
            show_frame_effdet(lmain, trigger, model)


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

    models_list = ["Yolov5", "EfficientDet D0"]

    chosen_model = tk.StringVar(frame)
    chosen_model.set("Yolov5")

    mod_menu = tk.OptionMenu(button_frame, chosen_model, *models_list)
    mod_menu.grid(row=1, column=0, padx=5, pady=5)

    window.mainloop()


def show_frame(lmain, trigger, model):
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
    lmain.after(20, lambda: show_frame(lmain, trigger, model))


def show_frame_effdet(lmain, trigger, model):
    _, frame = cap.read()
    if trigger.get() == "flip frame horizontally":
        frame = cv2.flip(frame, 1)

    ori_imgs, framed_imgs, framed_metas = preprocess_video(frame)
    x = torch.stack([torch.from_numpy(fi).to(device) for fi in framed_imgs], 0)
    x = x.to(torch.float32 if not use_float16 else torch.float16).permute(0, 3, 1, 2)
    with torch.no_grad():
        features, regression, classification, anchors = model(x)
        frame = postprocess(x, anchors, regression, classification, regressBoxes, clipBoxes, threshold, iou_threshold)
    frame = invert_affine(framed_metas, frame)
    frame = display(frame, ori_imgs)

    cv2image = cv2.cvtColor(frame, cv2.COLOR_BGR2RGBA)
    img = Image.fromarray(cv2image)
    img = img.resize((1280, 960), Image.ANTIALIAS)
    imgtk = ImageTk.PhotoImage(image=img)
    lmain.imgtk = imgtk
    lmain.configure(image=imgtk)
    lmain.after(20, lambda: show_frame_effdet(lmain, trigger, model))


def get_model(model_name="Yolov5"):
    if model_name == "Yolov5":

        # Model originally on GPU? If not cast to cpu it doesn't open display.
        if device == "cuda":
            model = torch.hub.load("ultralytics/yolov5", "yolov5l6", pretrained=True)
            model.half()
        else:
            model = torch.hub.load("ultralytics/yolov5", "yolov5s6", pretrained=True)
            model.to(device)

    else:
        compound_coef = int(model_name[-1])
        model = EfficientDetBackbone(compound_coef=compound_coef, num_classes=len(obj_list)).to(device)
        try:
            model.load_state_dict(torch.load(f'weights/efficientdet-d{compound_coef}.pth'))
        except FileNotFoundError:
            os.system(f"wget https://github.com/zylo117/Yet-Another-Efficient-Pytorch/releases/download/1.0/efficientdet-d{compound_coef}.pth -P weights/")
            model.load_state_dict(torch.load(f'weights/efficientdet-d{compound_coef}.pth'))
        model.requires_grad_(False)
        model.eval()

    return model


if __name__ == '__main__':
    cudnn.benchmark = True
    device = "cuda" if torch.cuda.is_available() else "cpu"

    regressBoxes = BBoxTransform()
    clipBoxes = ClipBoxes()
    threshold = 0.2
    iou_threshold = 0.2


    # Capture video frames
    cap = cv2.VideoCapture(0)

    tkinter_design()

    os.kill(os.getpid(), signal.SIGTERM)