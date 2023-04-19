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

train_cfg = dict(
    assigner=dict(
        type='MaxIoUAssigner',
        pos_iou_thr=0.5,
        neg_iou_thr=0.4,
        min_pos_iou=0,
        ignore_iof_thr=-1),
    allowed_border=-1,
    pos_weight=-1,
    debug=False)

test_cfg = dict(
    nms_pre=1000,
    min_bbox_size=0,
    score_thr=0.05,
    nms=dict(type='nms', iou_threshold=0.4),
    max_per_img=100)
train_cfg = Config(train_cfg)
test_cfg = Config(test_cfg)

def Retina_head():
 return mmdet.models.dense_heads.RetinaHead(num_classes=3,
                                            in_channels=256,
                                            stacked_convs=4,
                                            feat_channels=256,
                                            anchor_generator=dict(
                                                type='AnchorGenerator',
                                                scales = [3, 3.5, 4,4.3 ,4.5] , 
                                                ratios=[ 0.4,  0.5,0.6, 0.7,0.9,1.0 ],
                                                strides=[12, 24, 32]
                                                ),
                                            bbox_coder=dict(
                                                type='DeltaXYWHBBoxCoder',
                                                target_means=[
                                                    0.0, 0.0, 0.0, 0.0],
                                                target_stds=[1.0, 1.0, 1.0, 1.0]),
                                            loss_cls=dict(
                                                type='FocalLoss',
                                                use_sigmoid=True,
                                                gamma=2.0,
                                                alpha=0.25,
                                                loss_weight=0.7),
                                            loss_bbox=dict(type='L1Loss', loss_weight=0.3),
                                            train_cfg=train_cfg,
                                            test_cfg=test_cfg)
