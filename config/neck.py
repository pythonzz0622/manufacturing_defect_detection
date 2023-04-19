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

def Swin_L_neck(m_type):
    if m_type == 'fpn':
        return mmdet.models.necks.FPN(
            in_channels=[192, 384, 768, 1536],
            out_channels=256,
            start_level=1,
            add_extra_convs='on_output',
            num_outs=3)
    if m_type =='pafpn':
        return mmdet.models.necks.PAFPN(
                in_channels=[192, 384, 768, 1536],
                out_channels=256,
                start_level=1,
                add_extra_convs='on_output',
                num_outs=3)

def Resnet_152_neck(m_type):
    if m_type == 'fpn':
        return mmdet.models.necks.FPN(
            in_channels=[256, 512, 1024, 2048],
            out_channels=256,
            start_level=1,
            add_extra_convs='on_output',
            num_outs=3)
    if m_type =='pafpn':
        return mmdet.models.necks.PAFPN(
                in_channels=[256, 512, 1024, 2048],
                out_channels=256,
                start_level=1,
                add_extra_convs='on_output',
                num_outs=3)