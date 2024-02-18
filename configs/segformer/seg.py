_base_ = [
    '../_base_/models/segformer_mit-b0.py',   '../_base_/datasets/neu-seg.py',
    '../_base_/default_runtime.py', '../_base_/schedules/nextvit.py'
]
# model settings
model = dict(
    backbone=dict(
        embed_dims=64, num_heads=[1, 2, 5, 8], num_layers=[3, 4, 6, 3]),
    decode_head=dict(in_channels=[64, 128, 320, 512]))
