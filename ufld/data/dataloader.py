import os
import torch
import torchvision.transforms as transforms
from torch.utils.data import DataLoader, RandomSampler, SequentialSampler
import data.mytransforms as PrivateTransform
from data.constant import runway_row_anchor
from data.dataset import LaneClsDataset, LaneTestDataset


def get_train_loader(batch_size, data_root, griding_num, use_aux, num_lanes=2, rotate_angle=30):
    target_transform = transforms.Compose([
        PrivateTransform.FreeScaleMask((448, 448)),
        PrivateTransform.MaskToTensor(),
    ])
    segment_transform = transforms.Compose([
        PrivateTransform.FreeScaleMask((56, 56)),  # 由于插值后叠加，所以aux_seg特征图的尺寸相当于x2的尺寸，也就是原图大小的1/8（56）
        PrivateTransform.MaskToTensor(),
    ])
    img_transform = transforms.Compose([
        transforms.Resize((448, 448)),
        transforms.ToTensor(),
        transforms.Normalize((0.41846887, 0.44654263, 0.44974034), (0.23490292, 0.24692507, 0.26558167)),
    ])
    # common_transform = PrivateTransform.Compose2([
    #     PrivateTransform.RandomRotate(30),
    #     PrivateTransform.RandomUDoffsetLABEL(96),
    #     PrivateTransform.RandomLROffsetLABEL(96)
    # ])

    common_transform = PrivateTransform.RandomRotate(rotate_angle)

    # 只考虑CULane格式数据集。
    train_dataset = LaneClsDataset(data_root, os.path.join(data_root, 'list\\train_gt_resize_2.txt'),
                                   img_transform=img_transform,
                                   target_transform=target_transform,
                                   common_transform=common_transform,
                                   segment_transform=segment_transform,
                                   row_anchor=runway_row_anchor,  # 原为culane_row_anchor，改为runway_row_anchor
                                   griding_num=griding_num, use_aux=use_aux, num_lanes=num_lanes)
    cls_num_per_lane = 45  # TODO：也就是row_anchor个数？
    sampler = RandomSampler(train_dataset)

    train_loader = DataLoader(train_dataset, batch_size=batch_size, sampler=sampler, num_workers=0)

    return train_loader, cls_num_per_lane


def get_test_loader(batch_size, data_root, dataset, distributed):
    img_transforms = transforms.Compose([
        transforms.Resize((288, 800)),
        transforms.ToTensor(),
        transforms.Normalize((0.485, 0.456, 0.406), (0.229, 0.224, 0.225)),
    ])
    test_dataset = LaneTestDataset(data_root, os.path.join(data_root, 'list/test.txt'), img_transform=img_transforms)
    sampler = SequentialSampler(test_dataset)
    loader = DataLoader(test_dataset, batch_size=batch_size, sampler=sampler, num_workers=4)
    return loader
