import os.path as osp

import argparse
import torch
from typing import OrderedDict


def correct_unfold_reduction_order(x: torch.Tensor):
    out_channel, in_channel = x.shape
    x = x.reshape(out_channel, 4, in_channel // 4)
    x = x[:, [0, 2, 1, 3], :].transpose(1, 2).reshape(out_channel, in_channel)
    return x


def correct_unfold_norm_order(x):
    in_channel = x.shape[0]
    x = x.reshape(4, in_channel // 4)
    x = x[[0, 2, 1, 3], :].transpose(0, 1).reshape(in_channel)
    return x


def parse_args():
    parser = argparse.ArgumentParser()
    parser.add_argument('src_path', type=str)
    parser.add_argument('--dst_folder', type=str, default=None)
    parser.add_argument('-c', '--only-correct', action='store_true')

    args = parser.parse_args()
    return args


def main():
    args = parse_args()
    src_path = args.src_path
    dst_folder = args.dst_folder
    flag = args.only_correct

    raw_ckpt = torch.load(src_path)
    raw_ckpt = raw_ckpt.get('state_dict', raw_ckpt)
    raw_ckpt = raw_ckpt.get('model', raw_ckpt)
    new_ckpt = OrderedDict()
    if flag:
        for k, v in raw_ckpt.items():
            new_k = k
            if 'downsample' in k:
                if 'norm' in k:
                    v = correct_unfold_norm_order(v)
                elif 'reduction' in k:
                    v = correct_unfold_reduction_order(v)
            new_ckpt[new_k] = v
        if dst_folder is None:
            src_folder, filename = osp.split(src_path)
            dst_path = osp.join(src_folder,
                                filename.replace('.pth', '_converted.pth'))
            torch.save(new_ckpt, dst_path)
        else:
            src_folder, filename = osp.split(src_path)
            dst_path = osp.join(dst_folder,
                                filename.replace('.pth', '_converted.pth'))
            torch.save(new_ckpt, dst_path)


if __name__ == '__main__':
    main()
