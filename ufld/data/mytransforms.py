import numbers
import random
import numpy as np
from PIL import Image, ImageOps, ImageFilter
# from config import cfg
import torch
import pdb
import cv2


# ===============================img transforms============================

class Compose2(object):
    # transforms.Compose的二输入实现，每一个被Composed的变换都是针对image和mask同时进行的。call时还容许传递bbox参数
    def __init__(self, transforms):
        self.transforms = transforms

    def __call__(self, img, mask, bbox=None):
        if bbox is None:
            for t in self.transforms:
                img, mask = t(img, mask)
            return img, mask
        for t in self.transforms:
            img, mask, bbox = t(img, mask, bbox)
        return img, mask, bbox


# class FreeScale(object):
#     def __init__(self, size):
#         self.size = size  # (h, w)
#
#     def __call__(self, img, mask):
#         return img.resize((self.size[1], self.size[0]), Image.BILINEAR), mask.resize((self.size[1], self.size[0]),
#                                                                                      Image.NEAREST)


class FreeScaleMask(object):
    # 对输入的图像进行尺度变换。要求输入的图像必须是mask图像。
    def __init__(self, size):
        self.size = size

    def __call__(self, mask):
        return mask.resize((self.size[1], self.size[0]), Image.NEAREST)


# class Scale(object):
#     def __init__(self, size):
#         self.size = size
#
#     def __call__(self, img, mask):
#         if img.size != mask.size:
#             print(img.size)
#             print(mask.size)
#         assert img.size == mask.size
#         w, h = img.size
#         if (w <= h and w == self.size) or (h <= w and h == self.size):
#             return img, mask
#         if w < h:
#             ow = self.size
#             oh = int(self.size * h / w)
#             return img.resize((ow, oh), Image.BILINEAR), mask.resize((ow, oh), Image.NEAREST)
#         else:
#             oh = self.size
#             ow = int(self.size * w / h)
#             return img.resize((ow, oh), Image.BILINEAR), mask.resize((ow, oh), Image.NEAREST)


class RandomRotate(object):
    """Crops the given PIL.Image at a random location to have a region of
    the given size. size can be a tuple (target_height, target_width)
    or an integer, in which case the target will be of a square shape (size, size)
    """
    # 旨在同时对mask（label）和image进行旋转操作
    def __init__(self, angle):
        self.angle = angle

    def __call__(self, image, label):
        assert label is None or image.size == label.size

        angle = random.randint(0, self.angle * 2) - self.angle

        label = label.rotate(angle, resample=Image.NEAREST)
        image = image.rotate(angle, resample=Image.BILINEAR)

        return image, label


# ===============================label transforms============================

class DeNormalize(object):
    # 对一个张量进行反正规化，实际未使用
    def __init__(self, mean, std):
        self.mean = mean
        self.std = std

    def __call__(self, tensor):
        for t, m, s in zip(tensor, self.mean, self.std):
            t.mul_(s).add_(m)
        return tensor


class MaskToTensor(object):
    # 旨在将ndarray转换为tensor
    def __call__(self, img):
        return torch.from_numpy(np.array(img, dtype=np.int32)).long()


def find_start_pos(row_sample, start_line):
    # row_sample = row_sample.sort()
    # for i,r in enumerate(row_sample):
    #     if r >= start_line:
    #         return i

    l, r = 0, len(row_sample) - 1  # l初始化为0，r初始化为row_sample的高度-1
    while True:
        mid = int((l + r) / 2) # 求l和r的中间点mid
        if r - l == 1: # 如果r和l相邻（此时r在l下面一行），则返回r
            return r
        if row_sample[mid] < start_line:  # 如果start_line大于row_sample[mid]，则上方的l被替换为mid。
            l = mid
        if row_sample[mid] > start_line:  # 如果start_line小于，则下方的r被替换为mid
            r = mid
        if row_sample[mid] == start_line:  # 如果start_line等于，则返回这个mid值
            return mid


class RandomLROffsetLABEL(object):
    def __init__(self, max_offset):
        self.max_offset = max_offset

    def __call__(self, img, label):
        offset = np.random.randint(-self.max_offset, self.max_offset)
        w, h = img.size

        img = np.array(img)
        if offset > 0:
            img[:, offset:, :] = img[:, 0:w - offset, :]
            img[:, :offset, :] = 0
        if offset < 0:
            real_offset = -offset
            img[:, 0:w - real_offset, :] = img[:, real_offset:, :]
            img[:, w - real_offset:, :] = 0

        label = np.array(label)
        if offset > 0:
            label[:, offset:] = label[:, 0:w - offset]
            label[:, :offset] = 0
        if offset < 0:
            offset = -offset
            label[:, 0:w - offset] = label[:, offset:]
            label[:, w - offset:] = 0
        return Image.fromarray(img), Image.fromarray(label)


class RandomUDoffsetLABEL(object):
    def __init__(self, max_offset):
        self.max_offset = max_offset

    def __call__(self, img, label):
        offset = np.random.randint(-self.max_offset, self.max_offset)
        w, h = img.size

        img = np.array(img)
        if offset > 0:
            img[offset:, :, :] = img[0:h - offset, :, :]
            img[:offset, :, :] = 0
        if offset < 0:
            real_offset = -offset
            img[0:h - real_offset, :, :] = img[real_offset:, :, :]
            img[h - real_offset:, :, :] = 0

        label = np.array(label)
        if offset > 0:
            label[offset:, :] = label[0:h - offset, :]
            label[:offset, :] = 0
        if offset < 0:
            offset = -offset
            label[0:h - offset, :] = label[offset:, :]
            label[h - offset:, :] = 0
        return Image.fromarray(img), Image.fromarray(label)
