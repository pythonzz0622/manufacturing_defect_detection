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

parser = argparse.ArgumentParser()

parser.add_argument('--save_name' , help = 'retina-fpn-swin-l' , type = str)
parser.add_argument('--load_path' , help = './ckpts/data_retina_num_5000_epoch_100.pth' , type = str)
parser.add_argument('--epochs' , help = '100' , type = int)
parser.add_argument('--interval',  help = '100' , type = int)
parser.add_argument('--train_path' , help ='./dataset/train.json' , type =str)
parser.add_argument('--val_path', help = './dataset/test.json' , type =str)
args = parser.parse_args()

# set transformer ---------------------------------------------------------------
mean , std = (0.5, 0.5, 0.5) , (0.5, 0.5, 0.5)

train_transformer = A.Compose([
    A.Resize(512, 512),
    A.VerticalFlip(),
    A.RandomRotate90(),
    A.HorizontalFlip(),
    A.Normalize(mean=mean, std=std, max_pixel_value=255.0),
    ToTensorV2()], 
    bbox_params=A.BboxParams(format='coco', label_fields=['class_ids']))

val_transformer = A.Compose([
    A.Resize(height=512, width=512),
    A.Normalize(mean=mean, std=std, max_pixel_value=255.0),
    ToTensorV2(),],
    bbox_params=A.BboxParams(format='coco', label_fields=['class_ids']),)

# set config -------------------------------------------------------------------------
run = wandb.init(
    id = "retina-pafpn-resnet",
    project='SD_E&T_v4',
    notes="defect",
    entity = "gnu-ml-lab" , 
    mode="disabled"
    )

backbone = 'Rsenet152'
neck = 'Resnet_152_neck'
bbox_head = 'Retina_head'
neck_type = 'pafpn'
prefix_size =  3
save_name = args.save_name
logger = logger.create_logger(save_name)
load_path = args.load_path
os.makedirs(f'./ckpts/{save_name}' ,exist_ok= True)
os.makedirs(f'./result/{save_name}' ,exist_ok= True)
epochs = args.epochs
interval = args.interval
train_path = args.train_path
val_path = args.val_path
interval = 1

train_dataloader = import_data_loader( train_path, transformer=train_transformer , prefix_size= prefix_size)
val_dataset = customLoader.TestCustomDataset( val_path, transformer=val_transformer)
val_dataloader = DataLoader(dataset=val_dataset, batch_size=8, collate_fn=customLoader.collate_fn)
val_nums = len(val_dataloader)
cvt = tcvt.COCO_converter(val_path)

de_std , de_mean = tuple(std * 255 for std in std) , tuple(mean * 255 for mean in mean)


model = config.get_model(backbone_name = backbone, 
                                   neck_name = neck ,
                                    bbox_head_name =  bbox_head , neck_type=neck_type)

optimizer = optim.AdamW(model.parameters(), lr=0.0001 / 8, betas=(0.9, 0.999), weight_decay=0.05,)
lr_scheduler = CosineLRScheduler(
    optimizer,
    t_initial=(2000 - 500),
    lr_min=5e-6,
    warmup_lr_init=5e-7,
    warmup_t=1000,
    cycle_limit=20,
    t_in_epochs=False,
    warmup_prefix=True,
)



wandb.define_metric("train/step"); wandb.define_metric("train/*", step_metric="train/step"); 
wandb.define_metric("val/epoch");  wandb.define_metric("val/*", step_metric="val/epoch")

step = 1

if load_path: model.load_state_dict(torch.load(load_path))
model = model.cuda()

for epoch in range(1, epochs + 1):
    logger.info(f'epoch : {epoch}')
    cls_losses, bbox_losses, total_losses = 0, 0, 0
    
    for i, (img_metas, images, bboxes, labels) in enumerate(train_dataloader,1):
        if i  < 5:
            wandb.log({'train/bbox_debug' : utils.visualization.input_visual(images , bboxes , 8 , de_std , de_mean)} ,step = step )

        model.train()
        outputs , loss_bbox_total , loss_cls_total, total_loss = utils.iter_model(model , img_metas , images , bboxes , labels)
        cls_losses += loss_cls_total
        bbox_losses += loss_bbox_total
        total_losses += total_loss
        
        optimizer.zero_grad()
        total_loss.backward()
        optimizer.step()
        lr_scheduler.step_update(num_updates=step)
        step += 1
        if i % interval == 0:

            logger.info(f'[{i} / {len(train_dataloader)}]')
            logger.info(f'step : {i}\n \
                    loss_bbox : {bbox_losses / interval}  \
                    loss_cls_total : {cls_losses /interval}  \
                    total_loss : {total_losses / interval}')
            wandb.log({
                'train/step' : step , 
                'train/lr' : optimizer.param_groups[0]['lr'],
                "train/loss_bbox" : bbox_losses / interval,
                "train/loss_cls" : cls_losses / interval,
                "train/loss_total" : total_losses / interval})
            cls_losses, bbox_losses, total_losses = 0, 0, 0

    
    torch.save(model.state_dict(), f'./ckpts/{save_name}/epoch_{epoch}.pth')
    if epoch % interval == 0:
        cls_losses , bbox_losses , total_losses = 0, 0, 0
        preds_list = list()
        with torch.no_grad():
            model.eval()
            for j, (img_metas, images, bboxes, labels) in enumerate(val_dataloader):
                
                outputs , loss_bbox_total , loss_cls_total, total_loss = utils.iter_model(model , img_metas , images , bboxes , labels)
                cls_losses += loss_cls_total
                bbox_losses += loss_bbox_total
                preds = model.bbox_head.get_bboxes(
                    cls_scores=outputs[0],
                    bbox_preds=outputs[1],
                    img_metas=img_metas ,
                    rescale = True)
                
                preds = [(pred[0].cpu().numpy(),pred[1].cpu().numpy()) for pred in preds]
                preds_list.extend(zip([cvt.img_name_to_id[ os.path.basename(img_meta['filename'])] for img_meta in img_metas], preds))

            result = tcvt.preds_to_json(preds_list)
                

            with open(f'./result/{save_name}/result_{epoch}.json', "w") as json_file:
                json.dump(result, json_file, indent=4 , cls = tcvt.NpEncoder)

            try:
                utils.visualization.visual_plot(
                    save_name = save_name,
                    img_metas = img_metas ,
                    val_nums = val_nums,
                    epoch = epoch ,
                    bbox_losses= bbox_losses,
                    cls_losses= cls_losses)
                
            except: pass
            

            logger.info(f'step : {j}\n \
                          loss_bbox : {bbox_losses / val_nums} \
                          loss_cls_total : {cls_losses /val_nums}  \
                          total_loss : {(bbox_losses + cls_losses) / val_nums}')
    
    print(f"end epoch {epoch} {'-' * 30}")
 
