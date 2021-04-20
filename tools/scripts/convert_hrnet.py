import argparse
from collections import OrderedDict

import torch


def convert(src, dst):
    """Convert keys in detectron pretrained ResNet models to pytorch style."""
    # convert to pytorch style
    state_dict = OrderedDict()
    src_dict = torch.load(src)
    src_state_dict = src_dict.get('state_dict', src_dict)
    for k, v in src_state_dict.items():
        new_key = k.replace('model', 'backbone')
        if new_key.startswith('backbone.last_layer.0'):
            state_dict[new_key.replace('backbone.last_layer.0',
                                       'decode_head.convs.0.conv')] = v
        elif new_key.startswith('backbone.last_layer.1'):
            state_dict[new_key.replace('backbone.last_layer.1',
                                       'decode_head.convs.0.bn')] = v
        elif new_key.startswith('backbone.last_layer.3'):
            state_dict[new_key.replace('backbone.last_layer.3',
                                       'decode_head.conv_seg')] = v
        else:
            state_dict[new_key] = v

    # save checkpoint
    checkpoint = dict()
    checkpoint['state_dict'] = state_dict
    assert len(state_dict) == len(src_state_dict)
    checkpoint['meta'] = dict()
    torch.save(checkpoint, dst)


def main():
    parser = argparse.ArgumentParser(description='Convert model keys')
    parser.add_argument('src', help='src detectron model path')
    parser.add_argument('dst', help='save path')
    args = parser.parse_args()
    convert(args.src, args.dst)


if __name__ == '__main__':
    main()
