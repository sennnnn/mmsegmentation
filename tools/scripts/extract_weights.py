import argparse
import json
import os
import os.path as osp
import shutil


def collect_eval(work_dir):
    json_path_list = [
        osp.join(work_dir, x) for x in os.listdir(work_dir)
        if osp.splitext(x)[1] == '.json'
    ]

    cfg_path = [
        osp.join(work_dir, x) for x in os.listdir(work_dir)
        if osp.splitext(x)[1] == '.py'
    ][0]

    def process(x):
        x_head = int(
            osp.split(x)[1].replace('.log.json', '').split('_')[0][-2::])
        x_tail = int(osp.split(x)[1].replace('.log.json', '').split('_')[1])

        return x_head * 1000000 + x_tail

    sorted_json_path_list = sorted(json_path_list, key=process)

    info_dict = {}
    for json_path in json_path_list:
        with open(json_path, 'r') as fp:
            lines = fp.readlines()[1:]
            for line in lines:
                single_record = json.loads(line.strip())
                if single_record['mode'] == 'val':
                    info_dict[single_record['iter']] = single_record['mIoU']

    return info_dict, sorted_json_path_list[0], cfg_path


def main():
    parser = argparse.ArgumentParser()

    parser.add_argument(
        '-f', '--folder', type=str, help='Input the related work_dir path.')
    parser.add_argument(
        '-s',
        '--save-folder',
        type=str,
        default='unknown.pth',
        help='The output weights, config, log save folder.')

    args = parser.parse_args()
    eval_dict, json_src_path, cfg_src_path = collect_eval(args.folder)

    sorted_eval_res = sorted(
        eval_dict.items(), key=lambda x: x[0], reverse=True)

    max_eval_iter, max_eval_res = sorted_eval_res[0]

    print(f'Extracted from {args.folder}')
    print(f'Max evaluation iteration: {max_eval_iter}\n\
Max evaluation result: {max_eval_res}')

    task_name = osp.split(args.folder)[1]
    json_name = osp.split(json_src_path)[1]
    cfg_name = osp.split(cfg_src_path)[1]
    time_flag = json_name.replace('.log.json', '')

    dst_folder = osp.join(args.save_folder, task_name)
    pth_src_path = osp.join(args.folder, f'iter_{max_eval_iter}.pth')
    pth_dst_path = osp.join(dst_folder, f'{task_name}_{time_flag}.pth')
    json_dst_path = osp.join(dst_folder, task_name + '-' + json_name)
    cfg_dst_path = osp.join(dst_folder, cfg_name)

    if not osp.exists(dst_folder):
        os.makedirs(dst_folder, 0o777)

    shutil.copy(pth_src_path, pth_dst_path)
    shutil.copy(json_src_path, json_dst_path)
    shutil.copy(cfg_src_path, cfg_dst_path)

    os.system(f'python tools/publish_model.py {pth_dst_path} {pth_dst_path}')


if __name__ == '__main__':
    main()
