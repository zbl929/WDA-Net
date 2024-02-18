_base_ = [
    '../_base_/models/nextvit_fpn.py', '../_base_/datasets/zblwater_2class.py',
    '../_base_/default_runtime.py', '../_base_/schedules/nextvit.py'
]
model = dict(
	decode_head=dict(num_classes=4))

