"""
Show Frame utils.
Contains all the show_frame functions to be import into main function.
Created on: 04 Aout 2021.
"""
import numpy as np
import cv2
import re

import torch
import torchvision.transforms as T
from torch.backends import cudnn

from efficientdet.utils import BBoxTransform, ClipBoxes
from backbone import EfficientDetBackbone
from myutils.myutils import invert_affine, postprocess, torch_preprocess_video, display

import os
import signal

import tkinter as tk
from PIL import Image, ImageTk

# from neural_style.nsutils import
from neural_style.transformer_net import TransformerNet
from neural_style.vgg import Vgg16

use_float16 = True


#%%
def show_frame(lmain, trigger, model, cap):
    """
    PyTorch Yolov5 model call.
    """
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
    lmain.after(20, lambda: show_frame(lmain, trigger, model, cap))


#%%
regressBoxes = BBoxTransform()
clipBoxes = ClipBoxes()
threshold = 0.2
iou_threshold = 0.2

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


def show_frame_effdet(lmain, trigger, model, cap, device):
    """
    PyTorch EfficientDet model call.
    """
    _, frame = cap.read()
    if trigger.get() == "flip frame horizontally":
        frame = cv2.flip(frame, 1)

    ori_imgs, framed_imgs, framed_metas = torch_preprocess_video(frame, device=device)
    # x = torch.stack([torch.from_numpy(fi).to(device) for fi in framed_imgs], 0)
    # x = framed_imgs.unsqueeze(0).permute(0, 3, 1, 2)
    x = framed_imgs.to(device)
    x = x.to(torch.float32 if not use_float16 else torch.float16)
    with torch.no_grad():
        features, regression, classification, anchors = model(x)
        out = postprocess(x, anchors, regression, classification, regressBoxes, clipBoxes, threshold, iou_threshold)
    out = invert_affine([framed_metas], out)
    frame = display(out, [frame], obj_list, imshow=False)

    cv2image = cv2.cvtColor(frame, cv2.COLOR_BGR2RGBA)
    img = Image.fromarray(cv2image)
    img = img.resize((1280, 960), Image.ANTIALIAS)
    imgtk = ImageTk.PhotoImage(image=img)
    lmain.imgtk = imgtk
    lmain.configure(image=imgtk)
    lmain.after(5, lambda: show_frame_effdet(lmain, trigger, model, cap, device))


#%%
content_transform = T.Compose([
    T.ToTensor(),
])


def show_frame_fnst(lmain, trigger, model, cap, device):
    """
    PyTorch Fast Neural Style Transform (FNST) model call.
    """
    _, frame = cap.read()
    if trigger.get() == "flip frame horizontally":
        frame = cv2.flip(frame, 1)

    frame = content_transform(frame).unsqueeze(0).to(device)
    frame = frame.to(torch.float32 if not use_float16 else torch.float16)

    with torch.no_grad():
        frame = model(frame)

    frame = torch.round(frame[0].permute(1, 2, 0)).type(torch.uint8)
    frame = frame.cpu().numpy()
    cv2image = cv2.cvtColor(frame, cv2.COLOR_BGR2RGBA)
    img = Image.fromarray(cv2image)
    img = img.resize((1280, 960), Image.ANTIALIAS)
    imgtk = ImageTk.PhotoImage(image=img)
    lmain.imgtk = imgtk
    lmain.configure(image=imgtk)
    lmain.after(5, lambda: show_frame_fnst(lmain, trigger, model, cap, device))


#%%
def show_frame_edge(lmain, trigger, model, cap, use_cuda):
    _, frame = cap.read()
    if trigger.get() == "flip frame horizontally":
        frame = cv2.flip(frame, 1)

    results = model(frame, use_cuda=use_cuda)
    img = Image.fromarray(results * 255)
    img = img.resize((1280, 960), Image.ANTIALIAS)
    imgtk = ImageTk.PhotoImage(image=img)
    lmain.imgtk = imgtk
    lmain.configure(image=imgtk)
    lmain.after(20, lambda: show_frame_edge(lmain, trigger, model, cap, use_cuda))
