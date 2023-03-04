# yapf:disable
log_config = dict(
    # 15 mini batch iter 마다 log 남김
    interval=100,
    hooks=[
        dict(type='TextLoggerHook'),
        # dict(type='MMDetWandbHook',
        #      init_kwargs={'project': 'mmdetection'},
        #      interval=10,
        #      log_checkpoint=True,
        #      log_checkpoint_metadata=True,
        #      num_eval_images=100,
        #      bbox_score_thr=0.3)
    ])
# yapf:enable
custom_hooks = [
    dict(type='NumClassCheckHook')

]
find_unused_parameters = True
dist_params = dict(backend='nccl')
log_level = 'INFO'
load_from = None
resume_from = None
workflow = [('train', 1)]

# 10 epoch 마다 valdation
evaluation = dict(interval=10, metric='bbox')  # evaluation inverval settingd
checkpoint_config = dict(interval=10)
