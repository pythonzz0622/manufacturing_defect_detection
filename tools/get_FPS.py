import sys 
sys.path.append('/home/user304/users/jiwon/defect_detection')
from utils.customLoader import import_data_loader
import utils.customLoader as customLoader
import config
import albumentations as A
from albumentations.pytorch import ToTensorV2
import torch.optim as optim
import torch.nn as nn
from torchvision.ops import focal_loss
import torch
import matplotlib.pyplot as plt
import numpy as np
import cv2
import utils.type_converter  as tcvt
from torch.utils.data import DataLoader
import os
import utils
import json
import wandb
import time
import argparse
import utils.logger as logger
from timm.scheduler.cosine_lr import CosineLRScheduler
import random
backbone = 'Rsenet152'
in_channels = [256, 512, 1024, 2048] 
neck = 'PAFPN'
bbox_head = 'Retina_head'


model = config.get_model(backbone_name = backbone, 
                         neck_name = neck , 
                         bbox_head_name = bbox_head , 
                         in_channels=in_channels)
mean = (0.485, 0.456, 0.406)
std = (0.229, 0.224, 0.225)
val_transformer = A.Compose([
    A.Resize(height=512, width=512),
    A.Normalize(mean=mean, std=std,
                max_pixel_value=255.0),
    ToTensorV2(),
],
    bbox_params=A.BboxParams(format='coco', label_fields=['class_ids']),
)
device = torch.device('cuda')
val_dataset = customLoader.TestCustomDataset(
    './dataset/coco.json', transformer=val_transformer)
val_dataloader = DataLoader(dataset=val_dataset, batch_size=1,
                            collate_fn=customLoader.collate_fn)
if __name__=='__main__':
    model = model.cuda()
    model.eval()
    with torch.no_grad():
        for i, (img_metas, images, bboxes, labels) in enumerate(val_dataloader):
            images = images.cuda()
            bboxes = [bbox.cuda() for bbox in bboxes]
            labels = [label.cuda() for label in labels]
            if i  == 50:
                start = time.time()
            outputs = model(images)
            # preds = model.bbox_head.get_bboxes(
            #     cls_scores=outputs[0],
            #     bbox_preds=outputs[1],
            #     img_metas=img_metas ,
            #     rescale = True
            # )
            if i == 1050:
                end = time.time()
                break
        
        print(1000 / (end - start))