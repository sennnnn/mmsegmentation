import argparse
import json
import os
import os.path as osp


def parse_args():
    parser = argparse.ArgumentParser()
    parser.add_argument(
        'folder', type=str, help='The base folder of checkpoint.')

    args = parser.parse_args()

    return args


def main():
    args = parse_args()

    base_folder = args.folder
    json_path = [
        osp.join(base_folder, x) for x in os.listdir(base_folder)
        if osp.splitext(x)[1] == '.json'
    ][0]

    collect = []
    gpu_info = ''
    with open(json_path, 'r') as fp:
        lines = fp.readlines()
        for line in lines:
            info = json.loads(line)
            if 'env_info' in list(info.keys()):
                temp = info['env_info'].split('\n')
                gpu_info = [x for x in temp if 'GPU' in x][0]
            if 'mode' not in list(info.keys()):
                continue
            if info['iter'] <= 1000:
                collect.append(info['time'] - info['data_time'])
    avg_iter_time = sum(collect) / len(collect)
    print(gpu_info)
    print(f'Average iter time: {avg_iter_time}s')


if __name__ == '__main__':
    main()
