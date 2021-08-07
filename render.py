"""
Object detection algorithm.
"""
import sys

import numpy as np
import cv2
import re

import torch
from torch.backends import cudnn

from efficientdet.utils import BBoxTransform, ClipBoxes
from backbone import EfficientDetBackbone
from myutils.myutils import invert_affine, postprocess, torch_preprocess_video, display
from experimental.net_canny import canny

import os
import signal

import tkinter as tk
from PIL import Image, ImageTk

# from neural_style.nsutils import
from neural_style.transformer_net import TransformerNet
from neural_style.vgg import Vgg16

from show_frame_utils import *
from download_saved_models import dw_main

use_float16 = True


#%%
def tkinter_design():
    window = tk.Tk()
    window.wm_title("Object Detector")

    frame = tk.Frame(master=window, relief=tk.FLAT, height=250, width=250)
    frame.grid(row=0, column=0, padx=5, pady=5)

    window.columnconfigure(0, weight=1, minsize=50)
    window.rowconfigure(0, weight=1, minsize=50)

    def callback_start():
        # open_video()
        button.destroy()
        w.destroy()
        mod_menu.destroy()

        imageFrame = tk.Frame(master=window, width=1280, height=720)
        imageFrame.grid(row=0, column=0, padx=10, pady=2)
        lmain = tk.Label(imageFrame)
        lmain.grid(row=0, column=0)

        model = get_model(chosen_model.get(), fnst_choice.get())

        if chosen_model.get() == "Yolov5":
            show_frame(lmain, trigger, model, cap)
        elif chosen_model.get() == "Fast Neural Style Transfer":
            show_frame_fnst(lmain, trigger, model, cap, device)
        elif chosen_model.get() == "Edge Detection":
            use_cuda = True if device == "cuda" else False
            show_frame_edge(lmain, trigger, model, cap, device)
        else:
            show_frame_effdet(lmain, trigger, model, cap, device)


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

    models_list = ["Yolov5",
                   "EfficientDet D0",
                   "EfficientDet D4",
                   "EfficientDet D8",
                   "Fast Neural Style Transfer",
                   "Edge Detection"]
    # if device == "cuda": models_list += ["EfficientDet D4", "EfficientDet D8"]
    # models_list = sorted(models_list)

    chosen_model = tk.StringVar(frame)
    chosen_model.set("Yolov5")

    mod_menu = tk.OptionMenu(button_frame, chosen_model, *models_list)
    mod_menu.grid(row=1, column=0, padx=5, pady=5)

    #%%
    fnst_frame = tk.Frame(master=frame, relief=tk.FLAT)
    fnst_frame.grid(row=1, column=1, padx=5, pady=5)

    label1 = tk.Label(fnst_frame, text="For Fast Neural Style Transfer Only:")
    label1.grid(row=0, column=0)

    fnst_list = ["Rain Princess", "Candy", "Mosaic", "Udnie"]

    fnst_choice = tk.StringVar(frame)
    fnst_choice.set("Candy")

    fnst_menu = tk.OptionMenu(fnst_frame, fnst_choice, *fnst_list)
    fnst_menu.grid(row=1, column=0, padx=5, pady=5)

    window.mainloop()


#%%
def get_model(model_name="Yolov5", fnst_type=None):
    os_type = sys.platform

    if model_name == "Yolov5":

        # Model originally on GPU? If not cast to cpu it doesn't open display.
        if device == "cuda":
            model = torch.hub.load("ultralytics/yolov5", "yolov5l6", pretrained=True).to(device)
        else:
            model = torch.hub.load("ultralytics/yolov5", "yolov5s6", pretrained=True).to(device)

    elif model_name == "Fast Neural Style Transfer":

        try:
            if len(os.listdir("saved_models")) < 4:
                dw_main()
        except FileNotFoundError:
            dw_main()

        fnst_type = re.sub(" ", "_", fnst_type).lower()

        fnst_path = f"./saved_models/{fnst_type}.pth"

        def load_style(path):
            with torch.no_grad():
                model = TransformerNet()
                state_dict = torch.load(path)

                for k in list(state_dict.keys()):  # remove deprecation
                    if re.search(r'in\d+\.running_(mean|var)$', k): del state_dict[k]

                model.load_state_dict(state_dict)
                return model.to(device)

        model = load_style(fnst_path)

    elif re.match(r"Efficient", model_name):
        compound_coef = int(model_name[-1])
        model = EfficientDetBackbone(compound_coef=compound_coef, num_classes=len(obj_list), device=device)
        try:
            model.load_state_dict(torch.load(f'weights/efficientdet-d{compound_coef}.pth'))
        except FileNotFoundError:
            if compound_coef > 6: ver = 1.2
            else: ver = 1.0

            os.system(f"wget https://github.com/zylo117/Yet-Another-Efficient-Pytorch/releases/download/{ver}/efficientdet-d{compound_coef}.pth -P weights/")
            model.load_state_dict(torch.load(f'weights/efficientdet-d{compound_coef}.pth'))
        model.requires_grad_(False)
        model.eval()

    else:
        return canny

    if device == "cuda" and use_float16:
        model.half()

    return model.to(device)


#%%
if __name__ == '__main__':
    torch.set_grad_enabled(False)  # disable computation of gradient
    cudnn.enabled = True
    cudnn.benchmark = True

    device = "cuda" if torch.cuda.is_available() else "cpu"

    # Capture video frames
    cap = cv2.VideoCapture(0)

    tkinter_design()

    os.kill(os.getpid(), signal.SIGTERM)
