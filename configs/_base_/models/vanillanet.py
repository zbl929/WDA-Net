norm_cfg = dict(type='SyncBN', requires_grad=True)
checkpoint_file = '/home/zhangbulin/mmseg-0.23.0/pretrain/vanillanet_13.pth'
model = dict(
    type='EncoderDecoder',
    backbone=dict(
        _delete_=True,
        type='Vanillanet',
        act_num=3,  # enlarge act_num for better downstream performance
        dims=[128*4, 128*4, 256*4, 512*4, 512*4, 512*4, 512*4, 512*4, 512*4, 512*4, 1024*4, 1024*4],
        out_indices=[0, 1, 7, 9],
        strides=[1,2,2,1,1,1,1,1,1,2,1],
        init_cfg=dict(type='Pretrained', checkpoint=checkpoint_file),
        norm_cfg=norm_cfg,
        with_extra_norm=True,

       ),
    # neck=dict(type='FPN',
    #     in_channels=[128*4, 256*4, 512*4, 1024*4], out_channels=1024,num_outs=4),
    decode_head=dict(
        type='SegformerHead',
        in_channels=[128*4, 256*4, 512*4, 1024*4],
        in_index=[0, 1, 2, 3],
        channels=1024,
        dropout_ratio=0.1,
        num_classes=2,
        norm_cfg=norm_cfg,
        align_corners=False,
        # loss_decode=dict(
        #     type='CrossEntropyLoss', use_sigmoid=True, loss_weight=1.0)
        loss_decode=[
            dict(
                type='CrossEntropyLoss',
                loss_name='loss_ce',
                loss_weight=1.0,
                class_weight=[0.9373, 1.1055]),
            dict(
                type='FocalLoss',
                loss_name='loss_focal',
                loss_weight=1.0,
                class_weight=[0.9373, 1.1055]),
            dict(
                type='DiceLoss',
                loss_name='loss_dice',
                loss_weight=1.0,
                class_weight=[0.9373, 1.1055])]
    ),
    # model training and testing settings
    train_cfg=dict(),
    test_cfg=dict(mode='whole'))
