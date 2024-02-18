# model settings
norm_cfg = dict(type='SyncBN', requires_grad=True)
model = dict(
    type='EncoderDecoder',
    backbone=dict(
        type='EfficientViT_M4',
        # img_size=(512, 512),
        # pretrained="D:\CV_project\mmsegmentation-0.23.0\pretrain\mask_rcnn_efficientvit_m4_fpn_1x_coco.pth",
        frozen_stages=-1,
        ),
    neck=dict(
        type='EfficientViTFPN',
        in_channels=[128, 256, 384],
        out_channels=256,
        start_level=0,
        num_outs=5,
        num_extra_trans_convs=2,
    ),
    decode_head=dict(
        type='FPNHead',
        in_channels=[256, 256, 256, 256],
        in_index=[0, 1, 2, 3],
        feature_strides=[4, 8, 16, 32],
        channels=128,
        dropout_ratio=0.1,
        num_classes=2,
        norm_cfg=norm_cfg,
        align_corners=False,
        # loss_decode=dict(
        #     # type='CrossEntropyLoss', use_sigmoid=False, loss_weight=1.0)),
        #     type='CrossEntropyLoss', use_sigmoid=True, loss_weight=1.0)),
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
                class_weight=[0.9373, 1.1055])
        ]),
    # model training and testing settings
    train_cfg=dict(),
    test_cfg=dict(mode='whole'))


