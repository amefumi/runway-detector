#!/usr/bin/env python3
# -*- coding:utf-8 -*-
# Copyright (c) 2014-2021 Megvii Inc. All rights reserved.

import cv2
import matplotlib.pyplot
import numpy as np
import scipy.special
import torch
import torchvision.transforms as transforms
from PIL import Image
import torch
from ufld.data.constant import runway_row_anchor
from sklearn.linear_model import LinearRegression
from loguru import logger

__all__ = ["vis"]

runway_img_transforms = transforms.Compose([
    transforms.Resize((448, 448)),
    transforms.ToTensor(),
    transforms.Normalize((0.41846887, 0.44654263, 0.44974034), (0.23490292, 0.24692507, 0.26558167)),
])


def vis(img, boxes, scores, cls_ids, conf=0.5, class_names=None, obj_id=None, track_fps=-1, direction_model=None,
        text=None, lost_boxes=None):
    # 2021/12/27 添加了obj_id输入项，试图把plot_track和vis函数结合起来。
    for i in range(len(boxes)):
        box = boxes[i]
        cls_id = int(cls_ids[i])
        score = scores[i]
        if score < conf and obj_id is None:
            continue
        x0 = int(box[0])
        y0 = max(0, int(box[1]))
        x1 = int(box[2])
        y1 = int(box[3])

        color = (_COLORS[cls_id] * 255).astype(np.uint8).tolist()
        if obj_id is None:
            obj_text = '{}:{:.1f}%'.format(class_names[cls_id], score * 100)
        else:
            obj_text = '{}-ID:{}:{:.1f}%'.format(class_names[cls_id], obj_id[i], score * 100)
        txt_color = (0, 0, 0) if np.mean(_COLORS[cls_id]) > 0.5 else (255, 255, 255)
        font = cv2.FONT_HERSHEY_SIMPLEX

        txt_size = cv2.getTextSize(obj_text, font, 0.4, 1)[0]

        if direction_model is not None and cls_id == 0:  # 跑道的class id应该是0
            delta_x = int((x1 - x0) * 0.2)
            delta_y = int((y1 - y0) * 0.1)
            x0_d = max(0, x0 - delta_x)
            x1_d = min(len(img[0]) - 1, x1 + delta_x)
            # y0 = y0 - delta_y
            y1_d = min(len(img) - 1, y1 + delta_y)

            logger.info("Runway Detected at [{}, {}, {}, {}]".format(x0, y0, x1, y1))
            griding_num = 32
            cls_num_per_lane = 45

            runway_roi = img[y0:y1_d, x0_d:x1_d, :]  # 看YOLOX代码，这里的图像是cv2读入的BGR格式图像。
            runway_roi = cv2.cvtColor(runway_roi, cv2.COLOR_BGR2RGB)  # 根据测试，此处需要CVT！！
            cv2.imshow("slice", runway_roi)
            roi_height = len(runway_roi)
            roi_width = len(runway_roi[0])

            image_ndarray = Image.fromarray(runway_roi)
            image_tensor = runway_img_transforms(image_ndarray)
            image_tensor = image_tensor.unsqueeze(0)
            image_tensor = image_tensor.cuda()
            with torch.no_grad():
                direction_out = direction_model(image_tensor)
            col_sample = np.linspace(0, roi_width, griding_num)
            col_sample_w = col_sample[1] - col_sample[0]

            out_classification = direction_out[0].data.cpu().numpy()
            out_classification = out_classification[:, ::-1, :]
            prob = scipy.special.softmax(out_classification[:-1, :, :], axis=0)
            idx = np.arange(griding_num) + 1
            idx = idx.reshape(-1, 1, 1)
            loc = np.sum(prob * idx, axis=0)
            out_classification = np.argmax(out_classification, axis=0)
            loc[out_classification == griding_num] = 0
            out_classification = loc
            LR_model = LinearRegression()

            for j in range(out_classification.shape[1]):

                out_i = out_classification[:, j].reshape((-1, 1))
                out_index = out_i != 0
                if sum(out_index) != 0:
                    for k in range(out_classification.shape[0]):
                        if out_classification[k, j] > 0:
                            ppp = (round(out_classification[k, j] * col_sample_w -1 + x0_d),
                                   round(runway_row_anchor[cls_num_per_lane - 1 - k] * roi_height / 448 + y0))
                            cv2.circle(img, ppp, 3, (0, 100 + j * 120, 0), -1)

                    activate_anchor = np.array(runway_row_anchor).reshape((-1, 1))[out_index].reshape((-1, 1))
                    activate_out = out_i[out_index].reshape((-1, 1))
                    LR_model.fit(activate_anchor, activate_out)
                    out_lr = np.squeeze(LR_model.predict(np.array(runway_row_anchor).reshape((-1, 1))))
                    cv2.line(img,
                             (int(out_lr[0] * col_sample_w - 1 + x0_d),
                              int(runway_row_anchor[-1] * roi_height / 448 + y0)),
                             (int(out_lr[-1] * col_sample_w - 1 + x0_d),
                              int(runway_row_anchor[0] * roi_height / 448 + y0)),
                             (255, 0, 0), 2)
            cv2.rectangle(img, (x0_d, y0), (x1_d, y1_d), (0, 0, 255), 2)

        cv2.rectangle(img, (x0, y0), (x1, y1), color, 2)

        txt_bk_color = (_COLORS[cls_id] * 255 * 0.7).astype(np.uint8).tolist()
        cv2.rectangle(
            img,
            (x0, y0 + 1),
            (x0 + txt_size[0] + 1, y0 + int(1.5 * txt_size[1])),
            txt_bk_color,
            -1
        )
        cv2.putText(img, obj_text, (x0, y0 + txt_size[1]), font, 0.4, txt_color, thickness=1)
        cv2.putText(img, f'FPS:{track_fps}', (0, txt_size[1]), font, 0.4, txt_color, thickness=1)
        cv2.putText(img, f'{text}', (0, 3 * txt_size[1]), font, 0.4, txt_color, thickness=1)
    return img


_COLORS = np.array(
    [
        0.000, 0.447, 0.741,
        0.850, 0.325, 0.098,
        0.929, 0.694, 0.125,
        0.494, 0.184, 0.556,
        0.466, 0.674, 0.188,
        0.301, 0.745, 0.933,
        0.635, 0.078, 0.184,
        0.300, 0.300, 0.300,
        0.600, 0.600, 0.600,
        1.000, 0.000, 0.000,
        1.000, 0.500, 0.000,
        0.749, 0.749, 0.000,
        0.000, 1.000, 0.000,
        0.000, 0.000, 1.000,
        0.667, 0.000, 1.000,
        0.333, 0.333, 0.000,
        0.333, 0.667, 0.000,
        0.333, 1.000, 0.000,
        0.667, 0.333, 0.000,
        0.667, 0.667, 0.000,
        0.667, 1.000, 0.000,
        1.000, 0.333, 0.000,
        1.000, 0.667, 0.000,
        1.000, 1.000, 0.000,
        0.000, 0.333, 0.500,
        0.000, 0.667, 0.500,
        0.000, 1.000, 0.500,
        0.333, 0.000, 0.500,
        0.333, 0.333, 0.500,
        0.333, 0.667, 0.500,
        0.333, 1.000, 0.500,
        0.667, 0.000, 0.500,
        0.667, 0.333, 0.500,
        0.667, 0.667, 0.500,
        0.667, 1.000, 0.500,
        1.000, 0.000, 0.500,
        1.000, 0.333, 0.500,
        1.000, 0.667, 0.500,
        1.000, 1.000, 0.500,
        0.000, 0.333, 1.000,
        0.000, 0.667, 1.000,
        0.000, 1.000, 1.000,
        0.333, 0.000, 1.000,
        0.333, 0.333, 1.000,
        0.333, 0.667, 1.000,
        0.333, 1.000, 1.000,
        0.667, 0.000, 1.000,
        0.667, 0.333, 1.000,
        0.667, 0.667, 1.000,
        0.667, 1.000, 1.000,
        1.000, 0.000, 1.000,
        1.000, 0.333, 1.000,
        1.000, 0.667, 1.000,
        0.333, 0.000, 0.000,
        0.500, 0.000, 0.000,
        0.667, 0.000, 0.000,
        0.833, 0.000, 0.000,
        1.000, 0.000, 0.000,
        0.000, 0.167, 0.000,
        0.000, 0.333, 0.000,
        0.000, 0.500, 0.000,
        0.000, 0.667, 0.000,
        0.000, 0.833, 0.000,
        0.000, 1.000, 0.000,
        0.000, 0.000, 0.167,
        0.000, 0.000, 0.333,
        0.000, 0.000, 0.500,
        0.000, 0.000, 0.667,
        0.000, 0.000, 0.833,
        0.000, 0.000, 1.000,
        0.000, 0.000, 0.000,
        0.143, 0.143, 0.143,
        0.286, 0.286, 0.286,
        0.429, 0.429, 0.429,
        0.571, 0.571, 0.571,
        0.714, 0.714, 0.714,
        0.857, 0.857, 0.857,
        0.000, 0.447, 0.741,
        0.314, 0.717, 0.741,
        0.50, 0.5, 0
    ]
).astype(np.float32).reshape(-1, 3)
