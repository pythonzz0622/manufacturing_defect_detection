# yapf:disable
log_config = dict(
    # 15 mini batch iter 마다 log 남김
    interval=100,
    hooks=[
        dict(type='TextLoggerHook'),
        # dict(type='TensorboardLoggerHook')
    ])
# yapf:enable
custom_hooks = [
    dict(type='NumClassCheckHook')
]

dist_params = dict(backend='nccl')
log_level = 'INFO'
load_from = None
resume_from = None
workflow = [('train', 1)]

# 10 epoch 마다 valdation
evaluation = dict(interval=10, metric='bbox')  # evaluation inverval settingd
checkpoint_config = dict(interval=10)
