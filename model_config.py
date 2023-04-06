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

init_cfg = Config(dict(
type='Pretrained',
checkpoint= '/home/user304/users/jiwon/defect_detection/swin_large_patch4_window7_224_22k.pth'
))
backbone = mmdet.models.backbones.SwinTransformer(embed_dims=192,
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


neck = mmdet.models.necks.FPN(
    in_channels=[192, 384, 768, 1536],
    out_channels=256,
    start_level=1,
    add_extra_convs='on_output',
    num_outs=3)

train_cfg = dict(
    assigner=dict(
        type='MaxIoUAssigner',
        pos_iou_thr=0.5,
        neg_iou_thr=0.4,
        min_pos_iou=0,
        ignore_iof_thr=-1),
    allowed_border=0,
    pos_weight=-1,
    debug=False)
test_cfg = dict(
    nms_pre=1000,
    min_bbox_size=0,
    score_thr=0.3,
    nms=dict(type='nms', iou_threshold=0.5),
    max_per_img=100)
train_cfg = Config(train_cfg)
test_cfg = Config(test_cfg)

bbox_head = mmdet.models.dense_heads.RetinaHead(num_classes=3,
                                                in_channels=256,
                                                stacked_convs=4,
                                                feat_channels=256,
                                                anchor_generator=dict(
                                                    type='AnchorGenerator',
                                                    scales = [3,4,4.5] , 
                                                    ratios=[0.3, 0.4 ,0.6, 1.0],
                                                    strides=[12, 22, 32]),
                                                bbox_coder=dict(
                                                    type='DeltaXYWHBBoxCoder',
                                                    target_means=[
                                                        0.0, 0.0, 0.0, 0.0],
                                                    target_stds=[1.0, 1.0, 1.0, 1.0]),
                                                loss_cls=dict(
                                                    type='FocalLoss',
                                                    use_sigmoid=True,
                                                    gamma=1.0,
                                                    alpha=0.25,
                                                    loss_weight=1.3),
                                                loss_bbox=dict(type='L1Loss', loss_weight=0.7
                                                               ),
                                                train_cfg=train_cfg,
                                                test_cfg=test_cfg)


class retina_fpn_swin_l(nn.Module):
    def __init__(self):
        super().__init__()
        self.backbone = backbone
        self.backbone.init_weights()
        self.neck = neck
        self.bbox_head = bbox_head

    def forward(self, x):
        out = self.backbone(x)
        out = self.neck(out)
        out = self.bbox_head(out)
        return out


# if __name__ == "__main__":
#     inputs = []
#     model = retina_fpn_swin_l()
#     model.to('cuda')
#     # outputs = model(inputs)
