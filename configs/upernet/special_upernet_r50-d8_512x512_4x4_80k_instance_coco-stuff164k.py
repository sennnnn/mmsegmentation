_base_ = [
    '../_base_/models/special_upernet_r50-d8.py',
    '../_base_/datasets/instance_coco-stuff164k.py',
    '../_base_/default_runtime.py', '../_base_/schedules/schedule_80k.py'
]
model = dict(decode_head=dict(num_classes=172))
