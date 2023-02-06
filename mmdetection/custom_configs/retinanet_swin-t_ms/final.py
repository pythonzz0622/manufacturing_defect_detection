# final.py
_base_ = [
    'dataset.py',
    'schedule_1x.py',
    'default_runtime.py',
    'retinanet_swin-t.py'
]

lr_config = dict(warmup_iters=1000, step=[8, 11])
runner = dict(max_epochs=500)

data = dict(
    samples_per_gpu=24,
    workers_per_gpu=1)

optimizer = dict(
    _delete_=True,
    type='AdamW',
    lr=0.0001 / 8,
    betas=(0.9, 0.999),
    weight_decay=0.05,
    paramwise_cfg=dict(
        custom_keys={
            'absolute_pos_embed': dict(decay_mult=0.),
            'relative_position_bias_table': dict(decay_mult=0.),
            'norm': dict(decay_mult=0.)
        }))

checkpoint_config = dict(interval=10)

find_unused_parameters = True
