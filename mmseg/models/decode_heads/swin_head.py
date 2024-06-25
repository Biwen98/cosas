import torch
from torch import nn

from mmcv.cnn import build_norm_layer
from mmseg.registry import MODELS

from .decode_head import BaseDecodeHead

from ..utils.embed import PatchEmbed, PatchExpanding
from ..utils.swin_block import SwinBlockSequence


@MODELS.register_module()
class SwinHead(BaseDecodeHead):
    def __init__(self,
                 in_channels=[768, 768//2, 768//4, 768//8],
                 patch_size=4,
                 window_size=7,
                 mlp_ratio=4,
                 depths=(2, 2, 6, 2),
                 num_heads=(3, 6, 12, 24),
                 strides=(2, 2, 2, 4),
                 qkv_bias=True,
                 qk_scale=None,
                 patch_norm=True,
                 drop_rate=0.,
                 attn_drop_rate=0.,
                 drop_path_rate=0.1,
                 act_cfg=dict(type='GELU'),
                 norm_cfg=dict(type='LN'),
                 with_cp=False,
                 in_index=[0, 1, 2, 3],
                 **kwargs):
        super().__init__(in_channels, in_index=in_index, channels=in_channels[-1], input_transform='multiple_select', **kwargs)
        total_depth = sum(depths)
        dpr = [
            x.item() for x in torch.linspace(drop_path_rate, 0, total_depth)
        ]
        self.drop_before_pos = nn.Dropout(p=drop_rate)
        num_layers = len(depths)
        self.stages = nn.ModuleList()
        self.cates = nn.ModuleList()
        for i in range(num_layers):
            if i > 0:
                upsample = PatchExpanding(
                    in_channels=in_channels[i-1],
                    out_channels=in_channels[i],
                    stride=strides[i-1],
                    norm_cfg=norm_cfg if patch_norm else None,
                    init_cfg=None
                )
                cate = nn.Conv2d(in_channels[i-1], in_channels[i], kernel_size=1, stride=1, padding=0)
                self.cates.append(cate)
            else:
                upsample = None

            stage = SwinBlockSequence(
                embed_dims=in_channels[i],
                num_heads=num_heads[i],
                feedforward_channels=int(mlp_ratio * in_channels[i]),
                depth=depths[i],
                window_size=window_size,
                qkv_bias=qkv_bias,
                qk_scale=qk_scale,
                drop_rate=drop_rate,
                attn_drop_rate=attn_drop_rate,
                drop_path_rate=dpr[sum(depths[:i]):sum(depths[:i + 1])],
                sample=upsample,
                act_cfg=act_cfg,
                norm_cfg=norm_cfg,
                with_cp=with_cp,
                init_cfg=None)
            self.stages.append(stage)

        self.norms = nn.ModuleList()
        for i in range(len(self.in_channels)):
            layer = build_norm_layer(norm_cfg, self.in_channels[i])[1]
            self.norms.append(layer)

        self.patch_embed = PatchEmbed(
            in_channels=in_channels[-1],
            embed_dims=in_channels[-1],
            conv_type='deconv',
            kernel_size=patch_size,
            stride=strides[3],
            padding='corner',
            norm_cfg=norm_cfg if patch_norm else None,
            init_cfg=None)

    def forward(self, inputs):
        inputs = self._transform_inputs(inputs)[::-1]
        assert len(inputs) == len(self.stages)
        x = inputs[0]
        for i in range(len(self.stages)):
            if i == 0:
                hw_shape = x.shape[2:]
                x, hw_shape, out, out_hw_shape = self.stages[i](
                    x.reshape([x.shape[0], x.shape[1], hw_shape[0]*hw_shape[1]]).transpose(-2, -1),
                    hw_shape
                )
                x = self.norms[i](x)
                x = x.view(-1, *hw_shape, self.in_channels[i]).permute(0, 3, 1, 2).contiguous()
                #out = out.view(-1, *out_hw_shape, self.in_channels[i]).permute(0, 3, 1, 2).contiguous()
                #x = out

            else:
                x, hw_shape, out, out_hw_shape = self.stages[i](x, hw_shape)
                x = self.norms[i](x)
                x = x.view(-1, *hw_shape, self.in_channels[i]).permute(0, 3, 1, 2).contiguous()
                x = torch.cat((x, inputs[i]), dim=1)
                x = self.cates[i-1](x)
        x, hw_shape = self.patch_embed(x)
        x = self.drop_before_pos(x)
        x = x.view(-1, *hw_shape, self.in_channels[i]).permute(0, 3, 1, 2).contiguous()
        return self.cls_seg(x)





