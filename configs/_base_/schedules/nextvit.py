# optimizer
optimizer = dict(type='AdamW', lr=0.0001, weight_decay=0.01,
        # paramwise_cfg = dict(
        # custom_keys={
        #     'head': dict(lr_mult=2.)})
                 )
optimizer_config = dict(grad_clip=None)
# learning policy
lr_config = dict(policy='poly', power=0.9, min_lr=1e-6, by_epoch=False)
# runtime setti ngs, we use 8 gpu(total bs: 4*8=32) instead of 4 in mmsegmentation, so max_iters//2
runner = dict(type='IterBasedRunner', max_iters=10000)

checkpoint_config = dict(by_epoch=False, interval=4000)

evaluation = dict(interval=4000, metric='mIoU',save_best='mIoU',)

