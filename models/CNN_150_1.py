
"""
MobileNetV2 implementation used in
<Knowledge Distillation via Route Constrained Optimization>
"""


import torch.nn as nn
import math

import torch
import numpy as np
from torch.nn import init
from itertools import repeat
from torch.nn import functional as F
import collections.abc as container_abcs
from torch._jit_internal import Optional
from torch.nn.parameter import Parameter
from torch.nn.modules.module import Module
from PIL import Image
import torchvision.transforms as transforms
import matplotlib.pyplot as plt

from models import model_registry
__all__ = ['mobilenetv2_T_w', 'mobile_half']

BN = None


def _ntuple(n):
    def parse(x):
        if isinstance(x, container_abcs.Iterable):
            return x
        return tuple(repeat(x, n))

    return parse
_pair = _ntuple(2)



def conv_bn(inp, oup, stride):
    return nn.Sequential(
        nn.Conv2d(inp, oup, 3, stride, 1, bias=False),
        # nn.Conv2d(inp, oup, 3, stride, dilation=3, padding=3, bias=False),
        nn.BatchNorm2d(oup),
        nn.ReLU(inplace=True)
    )




def conv_1x1_bn(inp, oup):
    return nn.Sequential(
        nn.Conv2d(inp, oup, 1, 1, 0, bias=False),
        nn.BatchNorm2d(oup),
        nn.ReLU(inplace=True)
    )




class InvertedResidual(nn.Module):
    def __init__(self, inp, oup, stride, expand_ratio):
        super(InvertedResidual, self).__init__()
        self.blockname = None

        self.stride = stride
        assert stride in [1, 2]
        dw_stride = (1,stride)
        self.use_res_connect = self.stride == 1 and inp == oup
       # Conv2 = MaskConv2d(in_channels=inp * expand_ratio, out_channels=inp * expand_ratio, kernel_size=3, stride=stride, padding=1, groups=inp * expand_ratio, bias=False, conv_type=1,noise=False,normalization=True,offset=False,circle=True,sigma=5.0)
        self.conv = nn.Sequential(
            # pw
            nn.Conv2d(inp, inp * expand_ratio, 1, 1, 0, bias=False),
            nn.BatchNorm2d(inp * expand_ratio),
            nn.ReLU(inplace=True),
            # dw
            # Conv2,
            nn.Conv2d(inp * expand_ratio, inp * expand_ratio, 3, dw_stride, 1, groups=inp * expand_ratio, bias=False),
            nn.BatchNorm2d(inp * expand_ratio),
            nn.ReLU(inplace=True),
            # pw-linear
            nn.Conv2d(inp * expand_ratio, oup, 1, 1, 0, bias=False),
            nn.BatchNorm2d(oup),
        )
        self.names = ['0', '1', '2', '3', '4', '5', '6', '7']

    def forward(self, x):
        t = x
        if self.use_res_connect:
            return t + self.conv(x)
        else:
            return self.conv(x)


class MobileNetV2(nn.Module):
    """mobilenetV2"""
    def __init__(self, T,
                 feature_dim,
                 input_size=32,
                 width_mult=1.,
                 remove_avg=False):
        super(MobileNetV2, self).__init__()
        self.remove_avg = remove_avg

        # setting of inverted residual blocks  cifar 第二行s为1
        # self.interverted_residual_setting = [
        #     # t, c, n, s
        #     [1, 16, 1, 1],
        #     [T, 24, 2, 1],
        #     [T, 32, 3, 2],
        #     [T, 64, 4, 1],
        #     [T, 96, 3, 1],
        #     [T, 160, 3, 2],
        #     [T, 320, 1, 1],
        # ]

        self.interverted_residual_setting = [
            # t, c, n, s
            [1, 16, 1, 1],
            [T, 24, 2, 2],
            [T, 32, 3, 1],
            [T, 64, 4, 2],
            [T, 96, 3, 1],
            [T, 160, 3, 2],
            [T, 320, 1, 1],
        ]


        # building first layer

        input_channel = int(32 * width_mult)

        self.conv1 = conv_bn(2, input_channel, 2)

        # building inverted residual blocks
        self.blocks = nn.ModuleList([])
        for t, c, n, s in self.interverted_residual_setting:
            output_channel = int(c * width_mult)
            layers = []
            strides = [s] + [1] * (n - 1)
            for stride in strides:
                layers.append(
                    InvertedResidual(input_channel, output_channel, stride, t)
                )
                input_channel = output_channel
            self.blocks.append(nn.Sequential(*layers))

        self.last_channel = int(1280 * width_mult) if width_mult > 1.0 else 1280
        # self.last_channel = int(1280 * width_mult) if width_mult > 1.0 else int(c*width_mult)*4
        # self.conv2 = conv_1x1_bn(input_channel, self.last_channel)

        # building classifier
        self.classifier = nn.Sequential(
            nn.Dropout(0.2),
            nn.Linear(input_channel, feature_dim),
            # nn.Softmax()
        )

        # H = input_size // (32//2)
        # self.avgpool = nn.AvgPool2d(H, ceil_mode=True)
        self.avgpool = nn.AdaptiveAvgPool2d((1, 1))

        self._initialize_weights()
        print(T, width_mult)

    def get_bn_before_relu(self):
        bn1 = self.blocks[1][-1].conv[-1]
        bn2 = self.blocks[2][-1].conv[-1]
        bn3 = self.blocks[4][-1].conv[-1]
        bn4 = self.blocks[6][-1].conv[-1]
        return [bn1, bn2, bn3, bn4]

    def get_feat_modules(self):
        feat_m = nn.ModuleList([])
        feat_m.append(self.conv1)
        feat_m.append(self.blocks)
        return feat_m

    def forward(self, x, is_feat=False, preact=False):
        f00=x
        # x = self.act(self.prebn(self.pre(x)))

        # x = self.pre(x)
        # x = out +x
        x = x.permute(0, 3, 1, 2).contiguous()
        # print(x.shape)
        construct = x
        
        out = self.conv1(x)
        f0 = out

        out = self.blocks[0](out)
        out = self.blocks[1](out)
        f1 = out
        out = self.blocks[2](out)
        f2 = out
        out = self.blocks[3](out)
        out = self.blocks[4](out)
        f3 = out
        out = self.blocks[5](out)
        out = self.blocks[6](out)
        f4 = out

        # out = self.conv2(out)

        if not self.remove_avg:
            out = self.avgpool(out)
        out = out.view(out.size(0), -1)
        f5 = out
        out = self.classifier(out)

        if is_feat:
            return [f00,construct,f0, f1, f2, f3, f4, f5], out
        else:
            return out

    def _initialize_weights(self):
        for m in self.modules():
            if isinstance(m, nn.Conv2d):
                n = m.kernel_size[0] * m.kernel_size[1] * m.out_channels
                m.weight.data.normal_(0, math.sqrt(2. / n))
                if m.bias is not None:
                    m.bias.data.zero_()
            elif isinstance(m, nn.BatchNorm2d):
                m.weight.data.fill_(1)
                m.bias.data.zero_()
            elif isinstance(m, nn.Linear):
                n = m.weight.size(1)
                m.weight.data.normal_(0, 0.01)
                m.bias.data.zero_()


def mobilenetv2_T_w(T, W, feature_dim=100):
    model = MobileNetV2(T=T, feature_dim=feature_dim, width_mult=W)
    return model


def mobile_half(num_classes=2):
    return mobilenetv2_T_w(6, 0.5, num_classes)

model_registry.register_model("CNN_150", mobile_half)
from thop import profile
from fvcore.nn import FlopCountAnalysis, parameter_count_table
import tracemalloc
if __name__ == '__main__':
    x = torch.randn(1, 2, 5, 5)

    net = mobile_half(2)
    tracemalloc.start()
    with torch.no_grad():
        feats, logit = net(x, is_feat=True, preact=True)
    snapshot = tracemalloc.take_snapshot()
    top_stats = snapshot.statistics('lineno')

    # 找到峰值内存占用
    peak_memory = max(stat.size / (1024 * 1024) for stat in top_stats)
    print(f"Peak memory usage: {peak_memory:.2f} MB")

    # 停止内存分配跟踪
    tracemalloc.stop()
    for f in feats:
        print(f.shape, f.min().item())
    print(logit.shape)
    flops = FlopCountAnalysis(net,x)
    params = parameter_count_table(net)
    print(flops.total())
    print(params)
    flops, params = profile(net, (x,))
    print('flops: ', flops, 'params: ', params)
    for m in net.get_bn_before_relu():
        if isinstance(m, nn.BatchNorm2d):
            print('pass')
        else:
            print('warning')

