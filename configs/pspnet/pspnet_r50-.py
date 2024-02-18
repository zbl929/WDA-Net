_base_ = [
    '../_base_/models/pspnet_r50-d8.py', '../_base_/datasets/zblwater_2class.py',
    '../_base_/default_runtime.py',  '../_base_/schedules/nextvit.py'
]
model = dict(
    decode_head=dict(num_classes=3), auxiliary_head=dict(num_classes=3))
