import argparse
import os
import os.path as osp
import shutil
import subprocess

import torch


def process_checkpoint(in_file, out_file):
    checkpoint = torch.load(in_file, map_location='cpu')
    # remove optimizer for smaller file size
    if 'optimizer' in checkpoint:
        del checkpoint['optimizer']
    # if it is necessary to remove some sensitive data in checkpoint['meta'],
    # add the code here.
    torch.save(checkpoint, out_file)
    sha = subprocess.check_output(['sha256sum', out_file]).decode()
    final_file = out_file.rstrip('.pth') + '-{}.pth'.format(sha[:8])
    subprocess.Popen(['mv', out_file, final_file])


def construct(config_folder, ckpt_folder, save_folder, model_name, iters):
    item_list = [
        osp.splitext(x)[0] for x in os.listdir(config_folder)
        if x.upper() != 'README.MD'
    ]

    for i, item in enumerate(item_list):
        item_folder = osp.join(ckpt_folder, item)
        dst_folder = osp.join(save_folder, model_name, item)

        log_name = [
            x for x in os.listdir(item_folder) if osp.splitext(x)[1] == '.json'
        ][0]
        timepoint = log_name.replace('.log.json', '')

        ckpt_src_path = osp.join(item_folder, f'iter_{iters}.pth')
        ckpt_dst_path = osp.join(dst_folder, f'{item}_{timepoint}.pth')

        if not osp.exists(dst_folder):
            os.makedirs(dst_folder, 0o775)

        shutil.copy(ckpt_src_path, ckpt_dst_path)
        process_checkpoint(ckpt_dst_path, ckpt_dst_path)

        log_src_path = osp.join(item_folder, log_name)
        log_dst_path = osp.join(dst_folder, f'{item}_{log_name}')

        shutil.copy(log_src_path, log_dst_path)

        print(f'{i + 1}/{len(item_list)} {item} done!')


def parse_args():
    parser = argparse.ArgumentParser()
    parser.add_argument('config_folder', type=str, help='model config folder')
    parser.add_argument('ckpt_folder', type=str)
    parser.add_argument('save_folder', type=str)
    parser.add_argument('-m', '--model-name', type=str)
    parser.add_argument('-i', '--iters', type=int)
    args = parser.parse_args()
    return args


def main():
    args = parse_args()
    construct(args.config_folder, args.ckpt_folder, args.save_folder,
              args.model_name, args.iters)


if __name__ == '__main__':
    main()
