"""
Object detection algorithm.
"""
import numpy as np
import cv2

import torch


if __name__ == '__main__':

    model = torch.hub.load("ultralytics/yolov5", "yolov5l6", pretrained=True)

    cv2.namedWindow("preview")
    vc = cv2.VideoCapture(0)

    if vc.isOpened():
        rval, frame = vc.read()
    else: rval = False

    while rval:
        results = model(frame)
        results.display(render=True)
        cv2.imshow("preview", frame)
        rval, frame = vc.read()

        key = cv2.waitKey(20)
        if key == 27:  # ESC key.
            break

vc.release()
cv2.destroyWindow("preview")