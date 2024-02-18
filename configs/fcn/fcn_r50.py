# _base_ = [
#     '../_base_/models/fcn_r50-d8.py',  '../_base_/datasets/zblwater_2class.py',
#     '../_base_/default_runtime.py', '../_base_/schedules/nextvit.py'
# ]
_base_ = [
    '../_base_/models/fcn_r50-d8.py',  '../_base_/datasets/neu-seg.py',
    '../_base_/default_runtime.py', '../_base_/schedules/nextvit.py'
]
model = dict(
    decode_head=dict(num_classes=4), auxiliary_head=dict(num_classes=4))
