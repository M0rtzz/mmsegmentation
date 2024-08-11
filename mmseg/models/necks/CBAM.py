import torch
import torch.nn as nn
import torch.nn.functional as F
from mmcv.cnn import ConvModule
from mmengine.model import BaseModule

from mmseg.registry import MODELS
from ..utils import resize

class ChannelAttention(nn.Module):
    def __init__(self, in_planes, ratio=16):
        super(ChannelAttention, self).__init__()
        self.avg_pool = nn.AdaptiveAvgPool2d(1)
        self.max_pool = nn.AdaptiveMaxPool2d(1)

        self.fc1 = nn.Conv2d(in_planes, in_planes // ratio, 1, bias=False)
        self.relu1 = nn.ReLU()
        self.fc2 = nn.Conv2d(in_planes // ratio, in_planes, 1, bias=False)
        self.sigmoid = nn.Sigmoid()

    def forward(self, x):
        avg_out = self.fc2(self.relu1(self.fc1(self.avg_pool(x))))
        max_out = self.fc2(self.relu1(self.fc1(self.max_pool(x))))
        out = avg_out + max_out
        return self.sigmoid(out)


class SpatialAttention(nn.Module):
    def __init__(self, kernel_size=7):
        super(SpatialAttention, self).__init__()

        assert kernel_size in (3, 7), 'kernel size must be 3 or 7'
        padding = 3 if kernel_size == 7 else 1

        self.conv1 = nn.Conv2d(2, 1, kernel_size, padding=padding, bias=False)  # 7,3     3,1
        self.sigmoid = nn.Sigmoid()

    def forward(self, x):
        avg_out = torch.mean(x, dim=1, keepdim=True)
        max_out, _ = torch.max(x, dim=1, keepdim=True)
        x = torch.cat([avg_out, max_out], dim=1)
        x = self.conv1(x)
        return self.sigmoid(x)


class CBAM(nn.Module):
    def __init__(self, in_planes, ratio=16, kernel_size=7):
        super(CBAM, self).__init__()
        self.ca = ChannelAttention(in_planes, ratio)
        self.sa = SpatialAttention(kernel_size)

    def forward(self, x):
        out = x * self.ca(x)
        result = out * self.sa(out)
        return result


# 四层通道注意力机制
@MODELS.register_module()
class CBAM4(BaseModule):
    def __init__(self,
                 in_channels,
                 out_channels,
                 num_outs=4,
                 start_level=0,
                 end_level=-1,
                 conv_cfg=None,
                 norm_cfg=None

                 ):
        super().__init__()
        self.cbam0 = CBAM(in_channels[0])
        self.cbam1 = CBAM(in_channels[1])
        self.cbam2 = CBAM(in_channels[2])
        self.cbam3 = CBAM(in_channels[3])

    def forward(self, inputs):
        x0 = self.cbam0(inputs[0])
        x1 = self.cbam1(inputs[1])
        x2 = self.cbam2(inputs[2])
        x3 = self.cbam3(inputs[3])

        outs = []
        outs.append(x0)
        outs.append(x1)
        outs.append(x2)
        outs.append(x3)

        return tuple(outs)
