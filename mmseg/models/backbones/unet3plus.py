import warnings
import torch
import torch.nn as nn
from mmengine.model import BaseModule
from mmseg.registry import MODELS

class ConvBlock(nn.Module):
    def __init__(self, in_size, out_size, is_batchnorm, n=2, ks=3, stride=1, padding=1):
        super(ConvBlock, self).__init__()
        self.n = n
        self.ks = ks
        self.stride = stride
        self.padding = padding
        s = stride
        p = padding

        if is_batchnorm:
            for i in range(1, n + 1):
                conv = nn.Sequential(nn.Conv2d(in_size, out_size, ks, s, p),
                                     nn.BatchNorm2d(out_size), nn.ReLU(inplace=True), )
                setattr(self, 'conv%d' % i, conv)
                in_size = out_size
        else:
            for i in range(1, n + 1):
                conv = nn.Sequential(nn.Conv2d(in_size, out_size, ks, s, p), nn.ReLU(inplace=True), )
                setattr(self, 'conv%d' % i, conv)
                in_size = out_size

    def forward(self, inputs):
        x = inputs
        for i in range(1, self.n + 1):
            conv = getattr(self, 'conv%d' % i)
            x = conv(x)
        return x

@MODELS.register_module()
class UNet3Plus(BaseModule):
    def __init__(self,
                 in_channels=[3, 64, 128, 256, 512, 1024],
                # n_classes=1,
                 bilinear=True,
                 feature_scale=4,
                 is_deconv=True,
                 is_batchnorm=True,
                 pretrained=None,
                 init_cfg=None,
                 ):

        super().__init__(init_cfg)

        self.pretrained = pretrained
        assert not (init_cfg and pretrained), \
            'init_cfg and pretrained cannot be setting at the same time'
        if isinstance(pretrained, str):
            warnings.warn('DeprecationWarning: pretrained is a deprecated, '
                          'please use "init_cfg" instead')
            self.init_cfg = dict(type='Pretrained', checkpoint=pretrained)
        elif pretrained is None:
            if init_cfg is None:
                self.init_cfg = [
                    dict(type='Kaiming', layer='Conv2d'),
                    dict(
                        type='Constant',
                        val=1,
                        layer=['_BatchNorm', 'GroupNorm'])
                ]
        else:
            raise TypeError('pretrained must be a str or None')

        self.pretrained = pretrained
        assert not (init_cfg and pretrained), \
            'init_cfg and pretrained cannot be setting at the same time'
        if isinstance(pretrained, str):
            warnings.warn('DeprecationWarning: pretrained is a deprecated, '
                          'please use "init_cfg" instead')
            self.init_cfg = dict(type='Pretrained', checkpoint=pretrained)
        elif pretrained is None:
            if init_cfg is None:
                self.init_cfg = [
                    dict(type='Kaiming', layer='Conv2d'),
                    dict(
                        type='Constant',
                        val=1,
                        layer=['_BatchNorm', 'GroupNorm'])
                ]
        else:
            raise TypeError('pretrained must be a str or None')

        self.in_channels = in_channels
       # self.n_classes = n_classes
        self.bilinear = bilinear
        self.feature_scale = feature_scale
        self.is_deconv = is_deconv
        self.is_batchnorm = is_batchnorm
        # filters = [64, 128, 256, 512, 1024]
        self.layers = nn.ModuleList()
        for i in range(1, len(in_channels)):

            if i > 1:
                layer = torch.nn.Sequential(
                    nn.MaxPool2d(kernel_size=2),
                    ConvBlock(self.in_channels[i - 1], self.in_channels[i], self.is_batchnorm),
                )
            else:
                layer = ConvBlock(self.in_channels[i - 1], self.in_channels[i], self.is_batchnorm)
            self.layers.append(layer)

    def forward(self, inputs):
        outputs = []
        for layer in self.layers:
            inputs = layer(inputs)
            outputs.append(inputs)
        return outputs