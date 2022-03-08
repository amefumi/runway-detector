import cv2
import os

import numpy as np
import scipy.special
import torch
import torchvision.transforms as transforms
from loguru import logger
from PIL import Image

from data.constant import runway_row_anchor
from model.model import LaneSelectNet
from utils.common import merge_config

from sklearn.linear_model import LinearRegression

if __name__ == "__main__":

    args, cfg = merge_config()

    logger.info('Starting Image Demo...')
    assert cfg.backbone in ['18', '34', '50', '101', '152', '50next', '101next', '50wide', '101wide']

    cls_num_per_lane = 45  # 设置了45个row anchor

    net = LaneSelectNet(pretrained=False,
                        backbone=cfg.backbone,
                        net_para=(cfg.griding_num + 1, cls_num_per_lane, cfg.num_lanes),
                        use_aux=False).cuda()

    state_dict = torch.load(cfg.test_model, map_location='cpu')['model']
    compatible_state_dict = {}
    for k, v in state_dict.items():
        if 'module.' in k:
            compatible_state_dict[k[7:]] = v
        else:
            compatible_state_dict[k] = v

    net.load_state_dict(compatible_state_dict, strict=False)
    net.eval()

    img_transforms = transforms.Compose([
        transforms.Resize((448, 448)),
        transforms.ToTensor(),
        transforms.Normalize((0.41846887, 0.44654263, 0.44974034), (0.23490292, 0.24692507, 0.26558167)),
    ])

    img_list = os.listdir(os.path.join(cfg.data_root, 'CULaneDemo'))

    for input_image in img_list:
        frame = cv2.imread(os.path.join(cfg.data_root, 'CULaneDemo', input_image))
        img = cv2.cvtColor(frame, cv2.COLOR_BGR2RGB)
        height = len(img)
        width = len(img[0])
        logger.info('Loading image {}-H{}-W{}'.format(input_image, height, width))
        image_ndarray = Image.fromarray(img)
        image_tensor = img_transforms(image_ndarray)
        image_tensor = image_tensor.unsqueeze(0)
        image_tensor = image_tensor.cuda()
        with torch.no_grad():
            out = net(image_tensor)

        col_sample = np.linspace(0, 447, cfg.griding_num)
        col_sample_w = col_sample[1] - col_sample[0]

        out_classification = out[0].data.cpu().numpy()
        out_classification = out_classification[:, ::-1, :]
        prob = scipy.special.softmax(out_classification[:-1, :, :], axis=0)
        idx = np.arange(cfg.griding_num) + 1
        idx = idx.reshape(-1, 1, 1)
        loc = np.sum(prob * idx, axis=0)
        out_classification = np.argmax(out_classification, axis=0)
        loc[out_classification == cfg.griding_num] = 0
        out_classification = loc
        logger.debug(out_classification)

        frame = cv2.resize(frame, (448, 448))

        LR_model = LinearRegression()

        for i in range(out_classification.shape[1]):

            out_i = out_classification[:, i].reshape((-1, 1))
            out_index = out_i != 0
            if sum(out_index) != 0:
                if np.sum(out_classification[:, i] != 0) > 2:
                    for k in range(out_classification.shape[0]):
                        if out_classification[k, i] > 0:
                            ppp = (round(out_classification[k, i] * col_sample_w) - 1,
                                   round(runway_row_anchor[cls_num_per_lane - 1 - k]) - 1)
                            cv2.circle(frame, ppp, 5, (0, 255, 0), -1)

                activate_anchor = np.array(runway_row_anchor).reshape((-1, 1))[out_index].reshape((-1, 1))
                activate_out = out_i[out_index].reshape((-1, 1))
                LR_model.fit(activate_anchor, activate_out)
                out_lr = np.squeeze(LR_model.predict(np.array(runway_row_anchor).reshape((-1, 1))))
                print(out_lr)
                cv2.line(frame, (int(out_lr[0] * col_sample_w) - 1, int(runway_row_anchor[-1])),
                         (int(out_lr[-1] * col_sample_w) - 1, int(runway_row_anchor[0])), (255, 0, 0), 2)


                # cv2.imwrite(os.path.join(cfg.data_root, 'result', SImage), frame)  # 为了测试单张照片，不用视频形式输出。
        cv2.imshow('Test', frame)
        cv2.waitKey(0)
