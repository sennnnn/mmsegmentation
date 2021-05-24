import torch
import torch.nn as nn
from mmcv.cnn import ConvModule

from ..builder import HEADS
from .decode_head import BaseDecodeHead


@HEADS.register_module()
class SETRMLAHead(BaseDecodeHead):
    """Multi level feature aggretation head of SETR.

    MLA head of `SETR  <https://arxiv.org/pdf/2012.15840.pdf>`.

    Args:
        mlahead_channels (int): Channels of conv-conv-4x of multi-level feature
            aggregation. Default: 128.
    """

    def __init__(self, mla_channels=128, **kwargs):
        super(SETRMLAHead, self).__init__(
            input_transform='multiple_select', **kwargs)
        self.mla_channels = mla_channels

        num_inputs = len(self.in_channels)

        # Refer to self.cls_seg settings of BaseDecodeHead
        assert self.channels == num_inputs * mla_channels

        self.mlahead = nn.ModuleList()
        for i in range(num_inputs):
            self.mlahead.append(
                nn.Sequential(
                    ConvModule(
                        in_channels=self.in_channels[i],
                        out_channels=mla_channels,
                        kernel_size=3,
                        padding=1,
                        norm_cfg=self.norm_cfg,
                        act_cfg=self.act_cfg),
                    ConvModule(
                        in_channels=mla_channels,
                        out_channels=mla_channels,
                        kernel_size=3,
                        padding=1,
                        norm_cfg=self.norm_cfg,
                        act_cfg=self.act_cfg),
                    nn.Upsample(
                        scale_factor=4,
                        mode='bilinear',
                        align_corners=self.align_corners)))

    def forward(self, inputs):
        inputs = self._transform_inputs(inputs)
        outs = []
        for x, op in zip(inputs, self.mlahead):
            outs.append(op(x))
        out = torch.cat(outs, dim=1)
        out = self.cls_seg(out)
        return out
