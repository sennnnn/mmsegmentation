import argparse
from collections import OrderedDict

import torch


def convert(args):
    src_path = args.src_path
    dst_path = args.dst_path
    raw_ckpt = torch.load(src_path)
    new_ckpt = OrderedDict()
    raw_ckpt = raw_ckpt.get('state_dict', raw_ckpt)
    # if 'state_dict' in list(raw_ckpt.keys()):
    #     raw_ckpt = raw_ckpt['state_dict']
    backbone_convert(raw_ckpt, new_ckpt)

    head_selection = args.head_type
    if head_selection not in ['naive', 'pup', 'mla']:
        raise NotImplementedError
    else:
        if head_selection == 'naive':
            naive_head_convert(raw_ckpt, new_ckpt)
        elif head_selection == 'pup':
            pup_head_convert(raw_ckpt, new_ckpt)
        elif head_selection == 'mla':
            mla_head_convert(raw_ckpt, new_ckpt)

    torch.save(new_ckpt, dst_path)


def backbone_convert(raw_ckpt, new_ckpt):
    for k, v in raw_ckpt.items():
        # we move mla relative operation to mla head.
        if 'backbone.blocks' in k:
            new_ckpt[k] = v
        elif '_embed' in k:
            new_ckpt[k] = v
        elif 'cls_token' in k:
            new_ckpt[k] = v


def naive_head_convert(raw_ckpt, new_ckpt):
    for k, v in raw_ckpt.items():
        if 'conv_seg' in k:
            continue
        new_k = k
        if k.startswith('decode_head.conv_1'):
            new_k = new_k.replace('conv_1', 'conv_seg')
        elif k.startswith('decode_head.syncbn'):
            new_k = new_k.replace('sync', 'unified_')
        elif k.startswith('auxiliary_head'):
            if 'conv_1' in k:
                new_k = new_k.replace('conv_1', 'conv_seg')
            elif 'syncbn' in k:
                new_k = new_k.replace('sync', 'unified_')
        new_ckpt[new_k] = v


def pup_head_convert(raw_ckpt, new_ckpt):
    for k, v in raw_ckpt.items():
        if 'conv_seg' in k:
            continue
        new_k = k
        if k.startswith('decode_head.conv_4'):
            new_k = new_k.replace('conv_4', 'conv_seg')
        elif k.startswith('decode_head.syncbn'):
            new_k = new_k.replace('sync', 'unified_')
        elif k.startswith('auxiliary_head'):
            if 'conv_1' in k:
                new_k = new_k.replace('conv_1', 'conv_seg')
            elif 'syncbn' in k:
                new_k = new_k.replace('sync', 'unified_')
        new_ckpt[new_k] = v


def mla_head_convert(raw_ckpt, new_ckpt):
    for k, v in raw_ckpt.items():
        if 'conv_seg' in k:
            continue
        if k.startswith('backbone.mla'):
            new_ckpt[k.replace('backbone', 'decode_head')] = v
            for i in range(4):
                new_ckpt[k.replace('backbone', f'auxiliary_head.{i}')] = v
        elif k.startswith('backbone.norm_'):
            new_ckpt[k.replace('backbone.norm_', 'decode_head.norm.')] = v
            for i in range(4):
                new_ckpt[k.replace('backbone.norm_',
                                   f'auxiliary_head.{i}.norm.')] = v
        elif '.cls.' in k:
            new_ckpt[k.replace('.cls.', '.conv_seg.')] = v
        elif '.aux.' in k:
            new_ckpt[k.replace('.aux.', '.conv_seg.')] = v
        else:
            new_ckpt[k] = v


def parse_args():
    parser = argparse.ArgumentParser()
    parser.add_argument(
        '-t', '--head_type', type=str, help='Head type of SETR model.')
    parser.add_argument(
        'src_path',
        type=str,
        help='Path of official checkpoint of SETR model.')
    parser.add_argument(
        'dst_path', type=str, help='Save path of pytorch style checkpoint.')
    args = parser.parse_args()
    return args


def main():
    args = parse_args()
    convert(args)


if __name__ == '__main__':
    main()
