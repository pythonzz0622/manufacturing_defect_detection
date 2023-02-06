# dataset settings
dataset_type = 'CocoDataset'
# data_root = '../dataset/'  # dataset path
classes = ('over', 'under', 'non-welding')

img_norm_cfg = dict(
    mean=[123.675], std=[58.395], to_rgb=False)
train_pipeline = [
    dict(type='LoadImageFromFile'),
    dict(type='LoadAnnotations', with_bbox=True),
    dict(type='Resize', img_scale=[
         (512, 461), (512, 563)], multiscale_mode='range', keep_ratio=True),
    dict(type='RandomFlip', flip_ratio=0.5),
    dict(type='Normalize', **img_norm_cfg),
    dict(type='Pad', size_divisor=32),
    dict(type='DefaultFormatBundle'),
    dict(type='Collect', keys=['img', 'gt_bboxes', 'gt_labels']),
]
test_pipeline = [
    dict(type='LoadImageFromFile'),
    dict(
        type='MultiScaleFlipAug',
        img_scale=(512, 512),
        flip=False,
        transforms=[
            dict(type='Resize', keep_ratio=True),
            dict(type='RandomFlip'),
            dict(type='Normalize', **img_norm_cfg),
            dict(type='Pad', size_divisor=32),
            dict(type='ImageToTensor', keys=['img']),
            dict(type='Collect', keys=['img']),
        ])
]
data = dict(
    train=dict(
        type=dataset_type,
        ann_file='../info/train.json',
        img_prefix='../dataset/',
        classes=classes,
        pipeline=train_pipeline),
    val=dict(
        type=dataset_type,
        ann_file='../info/test.json',
        img_prefix='../dataset/',
        classes=classes,
        pipeline=test_pipeline),
    test=dict(
        type=dataset_type,
        ann_file='../info/test.json',
        img_prefix='../dataset/',
        classes=classes,
        pipeline=test_pipeline)
)

evaluation = dict(interval=10, metric='bbox')  # evaluation inverval settingd
