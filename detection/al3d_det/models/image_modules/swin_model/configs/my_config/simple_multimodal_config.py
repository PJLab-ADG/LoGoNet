_base_ = [
    'multi_modal_mask_rcnn_swin_fpn.py',
    '../_base_/datasets/coco_instance.py',
    '../_base_/schedules/schedule_1x.py', '../_base_/default_runtime.py'
]
# 用不同config中的模型定义替换 实现导入不同的模型
model = dict(
    backbone=dict(
        embed_dim=96,
        depths=[2, 2, 18, 2],
        num_heads=[3, 6, 12, 24],
        window_size=7,
        ape=False,
        drop_path_rate=0.2,
        patch_norm=True,
        use_checkpoint=False
    ),
    neck=dict(in_channels=[96, 192, 384, 768]))

img_norm_cfg = dict(
    mean=[123.675, 116.28, 103.53], std=[58.395, 57.12, 57.375], to_rgb=True)

simple_pipeline=[
    dict(type='LoadImageFromFile'),
    dict(type='Normalize', **img_norm_cfg),
    dict(type='Pad', size_divisor=32),
    dict(type='DefaultFormatBundle'),
    dict(type='Collect', keys=['img'],meta_keys=('filename', 'ori_filename', 'ori_shape',
                            'img_shape', 'pad_shape', 'scale_factor', 'img_norm_cfg')),
]

data = dict(train=dict(pipeline=simple_pipeline))