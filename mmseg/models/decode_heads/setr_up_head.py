import torch.nn as nn
from mmcv.cnn import ConvModule, build_norm_layer

from ..builder import HEADS
from ..utils import trunc_normal_
from .decode_head import BaseDecodeHead


@HEADS.register_module()
class SETRUPHead(BaseDecodeHead):
    """Naive upsampling head and Progressive upsampling head of SETR.

    Naive or PUP head of `SETR  <https://arxiv.org/pdf/2012.15840.pdf>`.

    Args:
        embed_dim (int): embedding dimension. Default: 1024.
        norm_layer (dict): Config dict for input normalization.
            Default: norm_layer=dict(type='LN', eps=1e-6, requires_grad=True).
        num_convs (int): Number of decoder convolutions. Default: 1.
        up_scale (int): The scale factor of interpolate. Default:4.
        kernel_size (int): The kernel size of convolution when decoding
            feature information from backbone. Default: 3.
    """

    def __init__(self,
                 norm_layer=dict(type='LN', eps=1e-6, requires_grad=True),
                 num_convs=1,
                 up_scale=4,
                 kernel_size=3,
                 **kwargs):

        assert kernel_size in [1, 3], 'kernel_size must be 1 or 3.'

        super(SETRUPHead, self).__init__(**kwargs)

        assert isinstance(self.in_channels, int)

        _, self.norm = build_norm_layer(norm_layer, self.in_channels)

        self.up = nn.ModuleList()
        in_channels = self.in_channels
        out_channels = self.channels
        for i in range(num_convs):
            self.up.append(
                nn.Sequential(
                    ConvModule(
                        in_channels=in_channels,
                        out_channels=out_channels,
                        kernel_size=kernel_size,
                        stride=1,
                        padding=int(kernel_size - 1) // 2,
                        norm_cfg=self.norm_cfg,
                        act_cfg=self.act_cfg),
                    nn.Upsample(
                        scale_factor=up_scale,
                        mode='bilinear',
                        align_corners=self.align_corners)))
            in_channels = out_channels

    def init_weights(self):
        for m in self.modules():
            if isinstance(m, nn.Linear):
                trunc_normal_(m.weight, std=.02)
                if isinstance(m, nn.Linear) and m.bias is not None:
                    nn.init.constant_(m.bias, 0)
            elif isinstance(m, nn.LayerNorm):
                nn.init.constant_(m.bias, 0)
                nn.init.constant_(m.weight, 1.0)

    def forward(self, x):
        x = self._transform_inputs(x)

        n, c, h, w = x.shape
        x = x.reshape(n, c, h * w).transpose(2, 1)
        x = self.norm(x)
        x = x.transpose(1, 2).reshape(n, c, h, w)

        for op in self.up:
            x = op(x)
        out = self.cls_seg(x)
        return out
