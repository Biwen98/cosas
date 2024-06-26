import warnings

import torch.nn
from mmengine.model import BaseModule
from mmseg.registry import MODELS
from ..utils import UpConvBlock
from ..utils.unet_block import BasicConvBlock


@MODELS.register_module()
class UNetNeck(BaseModule):
    def __init__(self,
                 in_channels=[1024, 512, 256, 128, 64],
                 strides=(1, 1, 1, 1, 1),
                 dec_num_convs=(2, 2, 2, 2),
                 upsamples=(True, True, True, True),
                 dec_dilations=(1, 1, 1, 1),
                 with_cp=False,
                 conv_cfg=None,
                 norm_cfg=dict(type='BN'),
                 act_cfg=dict(type='ReLU'),
                 upsample_cfg=dict(type='InterpConv'),
                 in_index=[0, 1, 2, 3, 4],
                 dcn=None,
                 plugins=None,
                 pretrained=None,
                 init_cfg=None
                 ):
        super().__init__(init_cfg=init_cfg)
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

        self.in_index = in_index
        num_stages = len(in_channels)
        assert dcn is None, 'Not implemented yet.'
        assert plugins is None, 'Not implemented yet.'
        assert len(strides) == num_stages, \
            'The length of strides should be equal to num_stages, '\
            f'while the strides is {strides}, the length of '\
            f'strides is {len(strides)}, and the num_stages is '\
            f'{num_stages}.'
        assert len(dec_num_convs) == (num_stages-1), \
            'The length of dec_num_convs should be equal to (num_stages-1), '\
            f'while the dec_num_convs is {dec_num_convs}, the length of '\
            f'dec_num_convs is {len(dec_num_convs)}, and the num_stages is '\
            f'{num_stages}.'
        assert len(upsamples) == (num_stages-1), \
            'The length of downsamples should be equal to (num_stages-1), '\
            f'while the downsamples is {upsamples}, the length of '\
            f'downsamples is {len(upsamples)}, and the num_stages is '\
            f'{num_stages}.'
        assert len(dec_dilations) == (num_stages-1), \
            'The length of dec_dilations should be equal to (num_stages-1), '\
            f'while the dec_dilations is {dec_dilations}, the length of '\
            f'dec_dilations is {len(dec_dilations)}, and the num_stages is '\
            f'{num_stages}.'

        assert isinstance(in_channels, (list, tuple))
        assert isinstance(in_index, (list, tuple))
        assert len(in_channels) == len(in_index)

        self.decoder = torch.nn.ModuleList()
        for i in range(num_stages):
            if i != (num_stages - 1):
                upsample = (strides[i] != 1 or upsamples[i])

                self.decoder.append(
                    UpConvBlock(
                    conv_block=BasicConvBlock,
                    in_channels=in_channels[i],
                    skip_channels=in_channels[i + 1],
                    out_channels=in_channels[i + 1],
                    num_convs=dec_num_convs[i],
                    stride=1,
                    dilation=dec_dilations[i],
                    with_cp=with_cp,
                    conv_cfg=conv_cfg,
                    norm_cfg=norm_cfg,
                    act_cfg=act_cfg,
                    upsample_cfg=upsample_cfg if upsample else None,
                    dcn=None,
                    plugins=None))

    def forward(self, inputs):
        inputs = [inputs[i] for i in self.in_index][::-1]
        x = inputs[0]

        dec_outs = [x]
        for i in range(0, len(self.decoder)):
            x = self.decoder[i](inputs[i + 1], x)
            dec_outs.append(x)
        return dec_outs
