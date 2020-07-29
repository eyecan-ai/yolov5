import numpy as np
import argparse

import torch.backends.cudnn as cudnn

from models.experimental import *
from models.experimental import attempt_load, check_img_size
from utils.datasets import *
from utils.datasets import LoadStreams, LoadImages
import os
import time
import random
import torch
import cv2
from pathlib import Path
from utils.utils import *
from utils.utils import (
    torch_utils, shutil, non_max_suppression, scale_coords,
    plot_one_box, plot_one_box_as_point, xyxy2xywh, strip_optimizer
)
from fpdf import FPDF
from itertools import count


class InferenceModel(object):

    def __init__(self,
                 img_size,
                 weights,
                 device='',
                 half=True,
                 conf_ths=0.4,
                 iou_ths=0.5,
                 augmented_inference=False,
                 classes_subset=None,
                 agnostic_nms=False):
        self._img_size = img_size
        self._weights = weights
        self._half = half
        self._augmented_inference = augmented_inference
        self._conf_ths = conf_ths
        self._iou_ths = iou_ths
        self._classes_subset = classes_subset
        self._agnostic_nms = agnostic_nms

        # Initialize
        self._device = torch_utils.select_device(device)

        self._model = attempt_load(self._weights, map_location=self._device)
        if half:
            self._model.half()

    def preprocess_image(self, image):
        # Padded resize
        img = letterbox(image, new_shape=self._img_size)[0]
        # Convert
        img = img[:, :, ::-1].transpose(2, 0, 1)  # BGR to RGB, to 3x416x416
        img = np.ascontiguousarray(img)

        img = torch.from_numpy(img).to(self._device)
        img = img.half() if self._half else img.float()  # uint8 to fp16/32
        img /= 255.0

        if img.ndimension() == 3:
            img = img.unsqueeze(0)

        return img

    def predict_raw(self, image, nms=True):

        image = self.preprocess_image(image)

        pred = self._model(
            image,
            augment=self._augmented_inference
        )[0]

        if nms:
            pred = non_max_suppression(
                pred,
                self._conf_ths,
                self._iou_ths,
                classes=self._classes_subset,
                agnostic=self._agnostic_nms
            )
        return image, pred

    def rescale(self, pred, processed_image, ref_image):
        rescaled = []
        for i, det in enumerate(pred):
            if det is not None and len(det):
                det[:, :4] = scale_coords(processed_image.shape[2:], det[:, :4], ref_image.shape).round()
                rescaled.append(det)
        return rescaled


def nearest(p, l2, min_dist=500, max_dist=600):
    lowest = np.inf
    target_p2 = None
    for p2 in l2:
        dist = np.linalg.norm(p - p2)
        if dist > min_dist and dist < max_dist and dist < lowest:
            target_p2 = p2
            lowest = dist
    return target_p2


def detect():

    output_folder = Path('/tmp/outputs')
    if not output_folder.exists():
        output_folder.mkdir(parents=True)
    output_counter = count()
    folder = Path('/home/daniele/Downloads/buffer_nastro_a_v')
    images_files = folder.glob('**/*.bmp')

    img_size = [512, 512]
    weights = '/home/daniele/work/workspace_python/yolov5/runs/exp10/weights/best.pt'
    inference_model = InferenceModel(
        img_size,
        weights,
        half=False,
        conf_ths=0.02,
        augmented_inference=True
    )

    colors = [(255, 100, 000), (0, 255, 100)]
    for image_file in images_files:
        if not image_file.is_file():
            continue
        ret = True
        print("Loading:", image_file)
        image = cv2.imread(str(image_file))  # cv2.imread('/media/daniele/Data/datasets/marchesini/scan_0/obj_006/20200725133041588600_45791657-ce6a-11ea-8df7-00d861df1197/00000.jpg')
        if ret:
            rescaled_image, pred = inference_model.predict_raw(image)
            rescaled_preds = inference_model.rescale(pred, rescaled_image, image)

            if rescaled_preds and len(rescaled_preds) > 0:

                output_preds = rescaled_preds[0].detach().cpu().numpy()
                print(output_preds)

                centers = {0: [], 1: []}
                for pred in output_preds:
                    *xyxy, conf, label = pred
                    x = xyxy
                    center = np.array([
                        (x[0] + x[2]) / 2,
                        (x[1] + x[3]) / 2
                    ])

                    * xyxy, conf, label = pred
                    label = int(label)
                    if label not in centers:
                        centers[label] = []
                    centers[label].append(center)

                for p in centers[0]:
                    p2 = nearest(p, centers[1])

                    cv2.circle(image, tuple(p.astype(int)), 10, colors[0], thickness=-1)
                    if p2 is not None:
                        cv2.circle(image, tuple(p2.astype(int)), 5, colors[1], thickness=-1)
                        cv2.arrowedLine(image,
                                        tuple(p.astype(int)),
                                        tuple(p2.astype(int)),
                                        colors[1],
                                        thickness=3,
                                        line_type=cv2.LINE_AA
                                        )

                # print(label)
                # text = str(f"{int(label)}_{conf:.2f}")
                # plot_one_box_as_point(xyxy, image, label='', color=colors[label], line_thickness=-1)

            cv2.imshow("image", image)

            c = cv2.waitKey(0)
            if c == ord('q'):
                break

            output_filename = output_folder / f'{next(output_counter)}.jpg'
            cv2.imwrite(str(output_filename), image)


detect()
