_base_ = [
    '../_base_/models/efficient.py', '../_base_/datasets/water.py',
    '../_base_/default_runtime.py', '../_base_/schedules/schedule_20k.py'
]


optimizer = dict(_delete_=True, type='AdamW', lr=0.0001, betas=(0.9, 0.999), weight_decay=0.05,
                 paramwise_cfg=dict(custom_keys={'attention_biases': dict(decay_mult=0.),
                                                 'attention_bias_idxs': dict(decay_mult=0.),
                                                 }))
# optimizer_config = dict(grad_clip=None)
# do not use mmdet version fp16
# fp16 = None
optimizer_config = dict(grad_clip=None)
# learning policy
