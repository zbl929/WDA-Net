# optimizer
optimizer = dict(type='AdamW', lr=0.0001, weight_decay=0.001)
optimizer_config=dict(_delete_=True, grad_clip=dict(max_norm=35, norm_type=2))

# learning policy
lr_config = dict(policy='poly', power=0.9, min_lr=1e-6, by_epoch=False)
# runtime settings, we use 8 gpu(total bs: 4*8=32) instead of 4 in mmsegmentation, so max_iters//2
runner = dict(type='IterBasedRunner', max_iters=40000)

checkpoint_config = dict(by_epoch=False, interval=4000)

evaluation = dict(interval=4000, metric='mIoU')