lr_config = dict(warmup_iters=1000, step=[27, 33])
runner = dict(type='EpochBasedRunner', max_epochs=100)
checkpoint_config = dict(interval=10)
optimizer_config = dict(grad_clip=None)
optimizer = dict(
    # _delete_=True,
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
lr_config = dict(
    policy='step',
    warmup='linear',
    warmup_iters=1000,
    warmup_ratio=0.001,
    step=[27, 33])
find_unused_parameters = True
