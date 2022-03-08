import torch
from PIL import Image
import os
import pdb
import numpy as np
from torch.utils.data import Dataset
from data.mytransforms import find_start_pos
import cv2

class LaneTestDataset(Dataset):
    def __init__(self, path, list_path, img_transform=None):
        super(LaneTestDataset, self).__init__()
        self.path = path
        self.img_transform = img_transform
        with open(list_path, 'r') as f:
            self.list = f.readlines()
        self.list = [l[1:] if l[0] == '/' else l for l in self.list]  # exclude the incorrect path prefix '/' of CULane

    def __getitem__(self, index):
        name = self.list[index].split()[0]
        img_path = os.path.join(self.path, name)
        img = Image.open(img_path)

        if self.img_transform is not None:
            img = self.img_transform(img)

        return img, name

    def __len__(self):
        return len(self.list)


class LaneClsDataset(Dataset):
    def __init__(self, path, list_path, img_transform=None, target_transform=None, common_transform=None,
                 griding_num=112, load_name=False, row_anchor=None, use_aux=False, segment_transform=None, num_lanes=2):
        super(LaneClsDataset, self).__init__()
        self.img_transform = img_transform
        self.target_transform = target_transform  # Target transform未使用
        self.segment_transform = segment_transform  # 如果使用aux方式，则需要返回一个经过Segment Transform变换的label图像
        self.common_transform = common_transform
        self.path = path
        self.griding_num = griding_num
        self.load_name = load_name
        self.use_aux = use_aux
        self.num_lanes = num_lanes
        self.fit_threshold = 6  # 对跑道执行拟合扩增的阈值，这里设置成6.但是实际上由于训练集中图片的特点，出现小于阈值的可能性很小。
        with open(list_path, 'r') as f:
            self.list = f.readlines()

        self.row_anchor = row_anchor
        self.row_anchor.sort()

    def __getitem__(self, index):
        list_item = self.list[index]  # list_item = "str_image_name str_label_name"
        item_info = list_item.split()
        image_name, label_name = item_info[0], item_info[1]
        if image_name[0] == '/':
            image_name = image_name[1:]
            label_name = label_name[1:]

        label_path = os.path.join(self.path, label_name)
        label = Image.open(label_path)
        # 对于CULane格式数据集而言，道路的标注还是以label图片格式存在的。所以制作数据集时要标注出来的JSON要转换为图片

        image_path = os.path.join(self.path, image_name)
        image = Image.open(image_path)

        if self.common_transform is not None:
            image, label = self.common_transform(image, label)  # 如果有common_transform，必定先进行common_transform
        lane_points = self.get_index(label)
        # get the coordinates of lanes at row anchors
        # lane_pts是在每个row_anchors处的边线的坐标

        classification_label = self.grid_points(lane_points, image.size[0])
        # make the coordinates to classification label
        # grid_pts函数将lane_pts坐标信息转换为分类标签格式信息。

        if self.img_transform is not None:
            image = self.img_transform(image)

        if self.use_aux:
            assert self.segment_transform is not None
            segmentation_image = self.segment_transform(label)
            return image, classification_label, segmentation_image

        if self.load_name:
            return image, classification_label, image_name

        return image, classification_label

    def __len__(self):
        return len(self.list)

    def grid_points(self, lane_points, image_width):
        # 把lane_points中具体的坐标分配给griding的坐标
        num_lanes, num_row_anchors, constant_2 = lane_points.shape
        assert constant_2 == 2

        col_sample = np.linspace(0, image_width - 1, self.griding_num)  # col_sample中有griding_num个元素。
        to_pts = np.zeros((num_row_anchors, num_lanes))  # (45, 2)

        for i in range(num_lanes):
            pti = lane_points[i, :, 1]
            to_pts[:, i] = np.asarray(
                [int(pt // (col_sample[1] - col_sample[0])) if pt != -1 else self.griding_num for pt in pti])
            # 如果用linspace分成griding_num个区间隔断，则最后一个像素对应的grid点实际上是griding_num-1，所以这里如果不存在，则分配到最后，是合理的！！
        return to_pts.astype(int)

    def get_index(self, label):
        image_width, image_height = label.size

        sample_temp = list(map(lambda x: int((x * 1.0 / 448) * image_height), self.row_anchor))
        # sample_temp是把所有可能的图片大小的row_anchor根据设定的row_anchor算出来。实际上没有什么必要。

        all_index = np.zeros((self.num_lanes, len(sample_temp), 2))
        # all_index: (车道线个数，row_anchor个数，2) => (2, 45, 2)
        for i, anchor_row in enumerate(sample_temp):
            # enumerate得到的i，r分别是（list内元素的序号，该元素的值）
            label_anchor_row = np.asarray(label)[int(round(anchor_row))]  # label_r是label图片在r行的数据。
            for lane_index in range(1, self.num_lanes + 1):
                lane_position = np.where(label_anchor_row == lane_index)[0]  # 在label_r中查找每一个车道线
                if len(lane_position) == 0:  # 如果label_anchor_row中没有找到任何车道线，那么要把all index矩阵里面的值设定为-1
                    all_index[lane_index - 1, i, 0] = anchor_row
                    all_index[lane_index - 1, i, 1] = -1
                else:
                    lane_position = np.mean(lane_position)
                    all_index[lane_index - 1, i, 0] = anchor_row
                    all_index[lane_index - 1, i, 1] = lane_position

        # data augmentation: extend the lane to the boundary of image

        all_index_extend = all_index.copy()
        # for i in range(self.num_lanes):
        #     if np.all(all_index_extend[i, :, 1] == -1):  # 对于i这一种类的车道线，all index矩阵中没有记录到车道在row anchor中的存在信息
        #         continue
        #     # if there is no lane
        #
        #     valid = all_index_extend[i, :, 1] != -1
        #     # get all valid lane points' index
        #     valid_lane_index = all_index_extend[i, valid, :]
        #     # get all valid lane points
        #     if valid_lane_index[-1, 0] == all_index_extend[0, -1, 0]:
        #         # if the last valid lane point's y-coordinate is already the last y-coordinate of all rows
        #         # this means this lane has reached the bottom boundary of the image
        #         # so we skip
        #         continue
        #     if len(valid_lane_index) < 6:
        #         continue
        #     # if the lane is too short to extend
        #
        #     valid_idx_half = valid_lane_index[len(valid_lane_index) // 2:, :]
        #     p = np.polyfit(valid_idx_half[:, 0], valid_idx_half[:, 1], deg=1)
        #     start_line = valid_idx_half[-1, 0]
        #     lane_position = find_start_pos(all_index_extend[i, :, 0], start_line) + 1
        #
        #     fitted = np.polyval(p, all_index_extend[i, lane_position:, 0])
        #     fitted = np.array([-1 if y < 0 or y > image_width - 1 else y for y in fitted])
        #
        #     assert np.all(all_index_extend[i, lane_position:, 1] == -1)
        #     all_index_extend[i, lane_position:, 1] = fitted
        # if -1 in all_index[:, :, 0]:
        #     pdb.set_trace()

        # 2022年2月21日：不再对跑道向下进行延申。
        return all_index_extend
