# row anchors are a series of pre-defined coordinates in image height to detect lanes
# the row anchors are defined according to the evaluation protocol of CULane and Tusimple
# since our method will resize the image to 288x800 for training, the row anchors are defined with the height of 288
# you can modify these row anchors according to your training image resolution
# row anchor是根据288*800的分辨率设计的，输入图目前变更为448*448，row anchor 应该进行改动。
# 对于TUSimple和CULane数据集而言，row anchor取决于跑道在图像中所占的位置。对本问题来说，应该改为全图都设置为anchor，然后插值确定轨迹线位置。

tusimple_row_anchor = [64, 68, 72, 76, 80, 84, 88, 92, 96, 100, 104, 108, 112, 116, 120, 124, 128, 132, 136, 140, 144,
                       148, 152, 156, 160, 164, 168, 172, 176, 180, 184, 188, 192, 196, 200, 204, 208, 212, 216, 220,
                       224, 228, 232, 236, 240, 244, 248, 252, 256, 260, 264, 268, 272, 276, 280, 284]

culane_row_anchor = [121, 131, 141, 150, 160, 170, 180, 189, 199, 209, 219, 228, 238, 248, 258, 267, 277, 287]

runway_row_anchor = [4, 14, 24, 34, 44, 54, 64, 74, 84, 94, 104, 114, 124, 134, 144, 154, 164, 174, 184, 194, 204, 214,
                     224, 234, 244, 254, 264, 274, 284, 294, 304, 314, 324, 334, 344, 354, 364, 374, 384, 394, 404, 414,
                     424, 434, 444]  # 共计45个行作为row anchor
