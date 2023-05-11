import sys 
sys.path.append('/home/user304/users/jiwon/defect_detection')
import sys
sys.setrecursionlimit(5000)
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
from ptflops import get_model_complexity_info
backbone = 'Rsenet152'
neck = 'FPN'
bbox_head = 'Retina_head'
in_channels = [192, 384, 768, 1536] 

model = config.get_model(backbone_name = backbone,  neck_name = neck , bbox_head_name = bbox_head , in_channels=in_channels)

with torch.cuda.device(0):
  macs, params = get_model_complexity_info(model, (3, 512, 512), as_strings=True,
                                           print_per_layer_stat=True, verbose=True)
  
  print(macs)
  print('-'* 50)
  print(params)