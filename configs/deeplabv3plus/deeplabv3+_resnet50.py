_base_ = [
    '../_base_/models/deeplabv3plus_r50-d8.py',
    '../_base_/datasets/zblwater_2class.py', '../_base_/default_runtime.py',
    '../_base_/schedules/nextvit.py'
]
model = dict(pretrained='open-mmlab://resnet50_v1c', backbone=dict(depth=50))