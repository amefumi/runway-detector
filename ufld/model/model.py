import torch
import torch.nn.functional as F
from .backbone import resnet
import numpy as np


class CBR(torch.nn.Module):
    """
    Conv BatchNorm ReLU合并的结构块。
    parameter:
    in_channels, out_channels, kernel_size, stride=1, padding=0, dilation=1, bias=False
    """

    def __init__(self, in_channels, out_channels, kernel_size, stride=1, padding=0, dilation=1, bias=False):
        super(CBR, self).__init__()
        self.conv = torch.nn.Conv2d(in_channels, out_channels, kernel_size, stride=stride, padding=padding,
                                    dilation=dilation, bias=bias)
        self.bn = torch.nn.BatchNorm2d(out_channels)
        self.relu = torch.nn.ReLU()

    def forward(self, x):
        x = self.conv(x)
        x = self.bn(x)
        x = self.relu(x)
        return x


class LaneSelectNet(torch.nn.Module):
    def __init__(self, image_size=(448, 448), pretrained=True, backbone='50', net_para=(57, 45, 2), use_aux=False):
        """
        跑道选择性识别网络。
        Args:
            image_size: 输入图片的尺寸，默认为(448, 448) - 主要参数A，由于ResNet特征图尺寸限制，两个数字都必须是32的倍数。
            pretrained: 预训练否，默认为True;
            backbone: 默认为ResNet50，backbone只取卷积层和池化层。;
            net_para: （横向分格（对应竖线）个数，纵向分格（对应row anchor）个数，车道个数（默认为2））- 主要参数B。
            use_aux: 是否使用辅助识别网络。训练时应该开启，推理时应该关闭。默认为False;
        """
        # 由于机场跑道终究不同于自动驾驶领域的问题，这里试着网络输入尺寸到（448，448）
        super(LaneSelectNet, self).__init__()

        self.image_size = image_size
        self.image_width = image_size[0]
        self.image_height = image_size[1]
        self.feature_para_num = (self.image_height // 32) * (self.image_width // 32) * 8
        self.net_parameter = net_para

        self.use_aux = use_aux
        self.total_dim = int(np.prod(net_para))

        # input : nchw, number, channels, heights, widths
        # output: (w+1) * sample_rows * 4
        self.model = resnet(backbone, pretrained=pretrained)

        if self.use_aux:
            self.aux_header2 = torch.nn.Sequential(
                CBR(128, 128, 3, padding=1) if backbone in ['34', '18'] else CBR(512, 128, 3, padding=1),
                CBR(128, 128, 3, padding=1),
                CBR(128, 128, 3, padding=1),
                CBR(128, 128, 3, padding=1),  # 重复的CBR有什么用吗？
            )
            self.aux_header3 = torch.nn.Sequential(
                CBR(256, 128, 3, padding=1) if backbone in ['34', '18'] else CBR(1024, 128, 3, padding=1),
                CBR(128, 128, 3, padding=1),
                CBR(128, 128, 3, padding=1),
            )
            self.aux_header4 = torch.nn.Sequential(
                CBR(512, 128, 3, padding=1) if backbone in ['34', '18'] else CBR(2048, 128, padding=1),
                CBR(128, 128, 3, padding=1),
            )
            self.aux_combine = torch.nn.Sequential(
                CBR(384, 256, 3, padding=2, dilation=2),
                CBR(256, 128, 3, padding=2, dilation=2),
                CBR(128, 128, 3, padding=2, dilation=2),
                CBR(128, 128, 3, padding=4, dilation=4),
                torch.nn.Conv2d(128, net_para[-1] + 1, 1)
                # output : n, num_of_lanes+1, h, w  输出通道数为跑道线个数+1，尺寸和输入尺寸height、weight相同。
            )
            init_weights(self.aux_header2, self.aux_header3, self.aux_header4, self.aux_combine)  # 初始化权重

        self.channel_adapt = torch.nn.Conv2d(512, 8, 1) if backbone in ['34', '18'] else torch.nn.Conv2d(2048, 8, 1)
        # 2022年2月11日：调大了线性部分输出通道数量。
        # ResNet50及之后网络的x4与50之前的x4通道数不同（但是分辨率相同），这里利用1*1卷积核统一了通道数为8。

        self.classifier = torch.nn.Sequential(
            torch.nn.Linear(self.feature_para_num, 2048),
            torch.nn.SiLU(),  # 提高了非线性的参与可能性？
            # 2022年2月11日换成了SiLU
            # torch.nn.Dropout(p=0.3),
            # 2022年2月21日取消了Dropout恢复成原代码格式
            torch.nn.Linear(2048, self.total_dim),
        )
        init_weights(self.classifier)

    def forward(self, x):
        x2, x3, x4 = self.model(x)
        # 特征图分别率分别为原图的1/8、1/16、1/32. 通道数分别为512、1024、2048（50及以后）；128、256、512（18、34）
        if self.use_aux:
            x2 = self.aux_header2(x2)
            x3 = self.aux_header3(x3)
            x3 = F.interpolate(x3, scale_factor=2, mode='bilinear')  # 这里对x3进行了插值，放大为2倍
            x4_aux = self.aux_header4(x4)
            x4_aux = F.interpolate(x4_aux, scale_factor=4, mode='bilinear')  # 这里对x4进行了插值，放大为4倍
            aux_seg = torch.cat([x2, x3, x4_aux], dim=1)
            aux_seg = self.aux_combine(aux_seg)
            # 由于插值后叠加，所以aux_seg特征图的尺寸相当于x2的尺寸，也就是原图大小的1/8
            # TODO：那么显而易见地，这样地aux_seg特征图对于问题地影响力很小，尤其是采用288*800输入时，得到的aux_seg显然比448*448输入有用得多。
        else:
            aux_seg = None

        x4 = self.channel_adapt(x4).view(-1, self.feature_para_num)  # x4是n，1，1，self.feature_para_num的张量
        group_cls = self.classifier(x4).view(-1, *self.net_parameter)

        if self.use_aux:
            return group_cls, aux_seg

        return group_cls


def init_weights(*models):
    # 初始化模型权重
    for model in models:
        exe_init_weights(model)


def exe_init_weights(model):
    if isinstance(model, list):  # 需要迭代地实现一些model的初始化，所以不能把这两个函数合并
        for mini_m in model:
            exe_init_weights(mini_m)
    else:
        if isinstance(model, torch.nn.Conv2d):
            torch.nn.init.kaiming_normal_(model.weight, nonlinearity='relu')  # TODO：这是什么意思，有什么好处？
            if model.bias is not None:
                torch.nn.init.constant_(model.bias, 0)
        elif isinstance(model, torch.nn.Linear):
            model.weight.data.normal_(0.0, std=0.01)
        elif isinstance(model, torch.nn.BatchNorm2d):
            torch.nn.init.constant_(model.weight, 1)
            torch.nn.init.constant_(model.bias, 0)
        elif isinstance(model, torch.nn.Module):
            for mini_m in model.children():
                exe_init_weights(mini_m)
        else:
            print('unknown module', model)
