optimizer = dict(
    #_delete_=True,
    constructor='LearningRateDecayOptimizerConstructor',
    type='AdamW',
    lr=1.3e-4,
    betas=(0.9, 0.999),
    weight_decay=0.05,
    paramwise_cfg={
        'decay_rate': 0.6,
        'decay_type': 'layer_wise',
        'num_layers': 6
    })
optimizer_config = dict(grad_clip=None)
lr_config = dict(
    policy='step',
    warmup='linear',
    warmup_iters=500,
    warmup_ratio=0.001,
    step=[15000, 20000])

runner = dict(type='IterBasedRunner', max_iters=20000)
checkpoint_config = dict(by_epoch=False, interval=2000)


evaluation = dict(interval=2000, metric='mIoU')
