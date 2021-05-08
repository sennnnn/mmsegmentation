import pytest
import torch

from mmseg.models.decode_heads import SETRMLAHead
from .utils import to_cuda


def test_setr_mla_head(capsys):

    with pytest.raises(AssertionError):
        # MLA requires input multiple stage feature information.
        SETRMLAHead(in_channels=32, channels=16, num_classes=19, in_index=1)

    with pytest.raises(AssertionError):
        # multiple in_indexs requires multiple in_channels.
        SETRMLAHead(
            in_channels=32, channels=16, num_classes=19, in_index=(0, 1, 2, 3))

        with pytest.raises(TypeError):
            # img_size must be int or tuple.
            SETRMLAHead(
                in_channels=(32, 32, 32, 32),
                channels=16,
                num_classes=19,
                in_index=(0, 1, 2, 3),
                img_size=[224, 224])

    # test inference of MLA head
    img_size = (32, 32)
    patch_size = 16
    head = SETRMLAHead(
        img_size=img_size,
        in_channels=(32, 32, 32, 32),
        channels=16,
        mla_channels=32,
        in_index=(0, 1, 2, 3),
        num_classes=19,
        norm_cfg=dict(type='BN'))

    h, w = img_size[0] // patch_size, img_size[1] // patch_size
    # Input NLC format feature information
    x = [
        torch.randn(1, h * w, 32),
        torch.randn(1, h * w, 32),
        torch.randn(1, h * w, 32),
        torch.randn(1, h * w, 32)
    ]
    if torch.cuda.is_available():
        head, x = to_cuda(head, x)
    out = head(x)
    assert out.shape == (1, head.num_classes, *img_size)

    # Input NCHW format feature information
    x = [
        torch.randn(1, 32, h, w),
        torch.randn(1, 32, h, w),
        torch.randn(1, 32, h, w),
        torch.randn(1, 32, h, w)
    ]
    if torch.cuda.is_available():
        head, x = to_cuda(head, x)
    out = head(x)
    assert out.shape == (1, head.num_classes, *img_size)