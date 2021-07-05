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
        env_info = json.loads(lines[0])['env_info'].split('\n')
        gpu_info = [x for x in env_info if 'GPU' in x][0]

        for line in lines[1:]:
            info = json.loads(line)
            collect.append(info['time'])
    avg_iter_time = (sum(collect) / len(collect)) / 50
    print(gpu_info)
    print(f'Average iter time: {avg_iter_time}s')


if __name__ == '__main__':
    main()
