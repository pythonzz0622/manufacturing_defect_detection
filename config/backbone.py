from mmcv import Config
# Copyright (c) OpenMMLab. All rights reserved.
import argparse
import copy
import os
import os.path as osp
import time
import warnings
import torch.nn as nn
import mmcv
import torch
import torch.distributed as dist
from mmcv import Config, DictAction
from mmcv.runner import get_dist_info, init_dist
from mmcv.utils import get_git_hash
import mmdet
import cv2
from pycocotools.coco import COCO
import torch
from mmdet import __version__
from mmdet.apis import init_random_seed, set_random_seed, train_detector
from mmdet.datasets import build_dataset
from mmdet.models import build_detector
from mmdet.utils import (collect_env, get_device, get_root_logger,
                         replace_cfg_vals, rfnext_init_model,
                         setup_multi_processes, update_data_root)
from mmdet.utils import (build_ddp, build_dp, compat_cfg,
                         find_latest_checkpoint, get_root_logger)

from mmdet.datasets import (build_dataloader, build_dataset,
                            replace_ImageToTensor)


def Swin_L():
    init_cfg = Config(dict(type='Pretrained', checkpoint= '/home/user304/users/jiwon/defect_detection/ckpts/pretrained/swin_large_patch4_window7_224_22k.pth' ))
    return mmdet.models.backbones.SwinTransformer(embed_dims=192,
                depths=[2, 2, 18, 2],
                num_heads=[6, 12, 24, 48],
                window_size=7,
                mlp_ratio=4,
                qkv_bias=True,
                qk_scale=None,
                drop_rate=0.0,
                attn_drop_rate=0.0,
                drop_path_rate=0.2,
                patch_norm=True,
                out_indices=(0, 1, 2, 3),
                with_cp=False,
                convert_weights=True,
                init_cfg=init_cfg)


def Rsenet50():
    return mmdet.models.backbones.ResNet(
                depth=152,
                num_stages=4,
                out_indices=(0, 1, 2, 3),
                frozen_stages=1,
                norm_cfg=dict(type='BN', requires_grad=True),
                norm_eval=True,
                style='pytorch',
                init_cfg=dict( type='Pretrained', checkpoint= 'torchvision://resnet152'))
