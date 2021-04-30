import pytest
import torch

from mmseg.models.decode_heads import SETRMLAAUXHead, SETRMLAHead
from .utils import to_cuda


def test_setr_mla_head(capsys):

    with pytest.raises(AssertionError):
        # MLA requires input multiple stage feature information.
        SETRMLAHead(in_channels=32, channels=16, num_classes=19, in_index=1)
        SETRMLAAUXHead(in_channels=32, channels=16, num_classes=19, in_index=1)

    # test inference of MLA head
    img_size = (32, 32)
    patch_size = 16
    head = SETRMLAHead(
        img_size=img_size,
        in_channels=(32, 32, 32, 32),
        channels=16,
        in_index=(0, 1, 2, 3),
        num_classes=19,
        embed_dim=32,
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

    # test inference of MLA AUX head
    img_size = (32, 32)
    patch_size = 16
    head = SETRMLAAUXHead(
        img_size=img_size,
        in_channels=(32, 32, 32, 32),
        channels=16,
        in_index=(0, 1, 2, 3),
        num_classes=19,
        embed_dim=32,
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