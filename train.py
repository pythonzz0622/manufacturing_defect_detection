from customLoader import import_data_loader
import customLoader
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

parser = argparse.ArgumentParser(description='coco_file을 가지고 train test를 나누는 script')

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
    id = "ret-fpn-swin-l",
    project='defect-detection',
    notes="defect",
    entity = "gnu-ml-lab" , 
    # mode="disabled"
    )
backbone = 'Swin_L'
neck = 'Swin_L_neck'
bbox_head = 'Retina_head'
prefix_size =  3
save_name = args.save_name
load_path = args.load_path
os.makedirs(f'./ckpts/{save_name}' ,exist_ok= True)
os.makedirs(f'./result/{save_name}' ,exist_ok= True)
epochs = args.epochs
interval = args.interval
train_path = args.train_path
val_path = args.val_path


train_dataloader = import_data_loader( train_path, transformer=train_transformer , prefix_size= prefix_size)
val_dataset = customLoader.TestCustomDataset( val_path, transformer=val_transformer)
val_dataloader = DataLoader(dataset=val_dataset, batch_size=8, collate_fn=customLoader.collate_fn)
val_nums = len(val_dataloader)
cvt = tcvt.COCO_converter(val_path)

de_std , de_mean = tuple(std * 255 for std in std) , tuple(mean * 255 for mean in mean)


model = config.get_model(backbone_name = backbone, 
                                   neck_name = neck ,
                                    bbox_head_name =  bbox_head)

optimizer = optim.AdamW(model.parameters(), lr=0.0001 / 8, betas=(0.9, 0.999), weight_decay=0.05,)
scheduler = optim.lr_scheduler.MultiStepLR(optimizer, milestones=[30,80], gamma=0.5)



wandb.define_metric("train/step"); wandb.define_metric("train/*", step_metric="train/step"); 
wandb.define_metric("val/epoch");  wandb.define_metric("val/*", step_metric="val/epoch")

step = 0

if load_path: model.load_state_dict(torch.load(load_path))
model = model.cuda()

for epoch in range(1, epochs + 1):
    print(f'epoch : {epoch}')
    cls_losses, bbox_losses, total_losses = 0, 0, 0
    
    for i, (img_metas, images, bboxes, labels) in enumerate(train_dataloader,1):
        step += i
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

        if i % interval == 0:
            scheduler.step()

            print(f'[{i} / {len(train_dataloader)}]')
            print(f'step : {i}\n \
                    loss_bbox : {bbox_losses / interval}  \
                    loss_cls_total : {cls_losses /interval}  \
                    total_loss : {total_losses / interval}')
            wandb.log({
                'train/step' : step , 
                'train/lr' : optimizer.param_groups[0]['lr'],
                "train/loss_bbox" : bbox_losses / interval,
                "train/loss_cls" : cls_losses / interval,
                "train/loss_total" : total_losses / interval
            })
            cls_losses, bbox_losses, total_losses = 0, 0, 0

    
    torch.save(model.state_dict(), f'./ckpts/{save_name}/epoch_{epoch}.pth')
    if epoch :
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
                    rescale = True
                )
                preds = [(pred[0].cpu().numpy(),pred[1].cpu().numpy())for pred in preds]
                preds_list.extend(zip( [cvt.img_name_to_id[ os.path.basename(img_meta['filename'])] for img_meta in img_metas], preds))

            result = tcvt.preds_to_json(preds_list)
                

            with open(f'./{result}/{save_name}/result_{epoch}.json', "w") as json_file:
                json.dump(result, json_file, indent=4 , cls = tcvt.NpEncoder)

            try:
                utils.visualization.visual_plot(
                    img_metas = img_metas ,
                    val_nums = val_nums,
                    epoch = epoch ,
                    bbox_losses= bbox_losses,
                    cls_losses= cls_losses)
                
            except: pass
            

            print(f'step : {j}\n \
                    loss_bbox : {bbox_losses / val_nums} \
                    loss_cls_total : {cls_losses /val_nums}  \
                    total_loss : {total_losses / val_nums}')
    
    print(f'end epoch {epoch}' +'-' * 30)

 