from customLoader import import_data_loader
import customLoader
from model_config import retina_fpn_swin_l
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
###################################### transformer ##############################
mean = (0.485, 0.456, 0.406)
std = (0.229, 0.224, 0.225)
transformer = A.Compose([
    A.Resize(512, 512),
    A.Flip(),
    A.OneOf([
        A.Rotate(limit=(0, 90), p=0.7),
        A.Compose([
            A.HorizontalFlip(always_apply=True),
            A.Rotate(limit=(-90, 0), p=0.7)
        ])
    ], p=0.7),
    A.Normalize(mean=mean, std=std,
                max_pixel_value=255.0),
    ToTensorV2()
], bbox_params=A.BboxParams(format='coco', label_fields=['class_ids']))

val_transformer = A.Compose([
    A.Resize(height=512, width=512),
    A.Normalize(mean=mean, std=std,
                max_pixel_value=255.0),
    ToTensorV2(),
],
    bbox_params=A.BboxParams(format='coco', label_fields=['class_ids']),
)

##########################################################################################
run = wandb.init(
  project="retina-fpn-swin-l",
  notes="test_experiment",
  entity = "gnu-ml-lab",
  tags=["baseline"]
)

device = torch.device('cuda')

train_dataset = customLoader.CustomDataset(
    './dataset/train.json', transformer=transformer)
train_dataloader = import_data_loader(
    './dataset/train.json', transformer=transformer)
val_dataset = customLoader.TestCustomDataset(
    './dataset/test.json', transformer=val_transformer)
val_dataloader = DataLoader(dataset=val_dataset, batch_size=8,
                            collate_fn=customLoader.collate_fn)
cvt = tcvt.COCO_converter('./dataset/test.json')

de_std = tuple(std * 255 for std in std)
de_mean = tuple(mean * 255 for mean in mean)


model = retina_fpn_swin_l()

optimizer = optim.AdamW(model.parameters(),
                        lr=0.0001 / 8,
                        betas=(0.9, 0.999),
                        weight_decay=0.05,
                        )

model = model.cuda()
# model.load_state_dict(torch.load('./test3.pth'))

# wandb.define_metric("train/step")
wandb.define_metric("val/epoch")
# wandb.define_metric("train/*", step_metric="train/step")
wandb.define_metric("val/*", step_metric="val/epoch")
epochs = 50
cls_losses = 0
bbox_losses = 0
total_losses = 0
step = 0

for epoch in range(1, epochs + 1):
    print(f'epoch : {epoch}')
    for i, (img_metas, images, bboxes, labels) in enumerate(train_dataloader):
        step += i
        images = images.cuda()
        model.train()

        bboxes = [bbox.cuda() for bbox in bboxes]
        labels = [label.cuda() for label in labels]
        outputs = model(images)
        if i ==1: start = time.time()
        losses = model.bbox_head.loss(
            cls_scores=outputs[0],
            bbox_preds=outputs[1],
            gt_bboxes=bboxes,
            gt_labels=labels,
            img_metas=img_metas
        )
        if i==1:
            end = time.time()
            delta = end - start 
            eta = delta * len(train_dataloader)* (epochs - epoch)
            utils.type_converter.sec_to_format(eta)
        loss_cls_total = losses['loss_cls'][0] + losses['loss_cls'][1] + losses['loss_cls'][2]
        loss_bbox_total = losses['loss_bbox'][0] +  losses['loss_bbox'][1] + losses['loss_bbox'][2]
        total_loss = loss_cls_total + loss_bbox_total
        total_loss.backward()
        optimizer.step()
        optimizer.zero_grad()

        cls_losses += loss_cls_total
        bbox_losses += loss_bbox_total
        total_losses += total_loss

        if i % 100 == 0 and i >1 :
            print(f'[{i} / {len(train_dataloader)}]')
            print(
                f'step : {i}\nloss_bbox : {bbox_losses / 100}  loss_cls_total : {cls_losses /100}  total_loss : {total_losses / 100}')
            train_log_dict = {
                'train/step' : step , 
                "train/loss_bbox" : bbox_losses / 100,
                "train/loss_cls" : cls_losses / 100,
                "train/loss_total" : total_losses / 100
            }
            wandb.log(train_log_dict)
            cls_losses = 0
            bbox_losses = 0
            total_losses = 0
    # if epoch % 10 == 0:
    if epoch % 5 == 0:
        cls_losses = 0
        bbox_losses = 0
        total_losses = 0

        preds_list = []
        with torch.no_grad():
            model.eval()
            for j, (img_metas, images, bboxes, labels) in enumerate(val_dataloader):

                images = images.cuda()
                bboxes = [bbox.cuda() for bbox in bboxes]
                labels = [label.cuda() for label in labels]
                outputs = model(images)
                losses = model.bbox_head.loss(
                    cls_scores=outputs[0],
                    bbox_preds=outputs[1],
                    gt_bboxes=bboxes,
                    gt_labels=labels,
                    img_metas=img_metas
                )
                loss_cls_total = losses['loss_cls'][0] + \
                    losses['loss_cls'][1] + losses['loss_cls'][2]
                loss_bbox_total = losses['loss_bbox'][0] + \
                    losses['loss_bbox'][1] + losses['loss_bbox'][2]
                total_loss = loss_cls_total + loss_bbox_total
                cls_losses += loss_cls_total
                bbox_losses += loss_bbox_total
                total_losses += total_loss

                preds = model.bbox_head.get_bboxes(
                    cls_scores=outputs[0],
                    bbox_preds=outputs[1],
                    img_metas=img_metas ,
                    rescale = True
                )
                preds = [(pred[0].cpu().numpy(),pred[1].cpu().numpy())for pred in preds]
                preds_list.extend(zip( [cvt.img_name_to_id[ os.path.basename(img_meta['filename'])] for img_meta in img_metas], preds))
                result = [{'image_id' : pred_list[0],
                            'bbox' : tcvt.cv2_to_coco(pred_list[1][0][i][:-1].tolist() , dim =1),
                            'score' : pred_list[1][0][i][-1],
                            'category_id' : pred_list[1][1][i]} for pred_list in preds_list  for i in range(len(pred_list[1][1]))]
                

            with open('result.json', "w") as json_file:
                json.dump(result, json_file, indent=4 , cls = tcvt.NpEncoder)

            visualtool = utils.visualization.VisualTool(coco_gt_path = './dataset/test.json' , coco_pred_path='./result.json' )
            PR_plot = visualtool.PR_plot()
            R_plot , P_plot , F1_plot = visualtool.conf_vs()
            coco_cvt = utils.type_converter.COCO_converter('./dataset/test.json' , 'result.json')
            mAP = coco_cvt.results['precision'][0, :, :, 0, -1].mean()
            cat1_precision ,cat1_recall  = coco_cvt.PR(conf_score=0.5, catId = 1)
            cat2_precision ,cat2_recall = coco_cvt.PR(conf_score=0.5, catId = 2)
            val_log_dict = {
            "val/epoch" : epoch,
            "val/loss_bbox" : bbox_losses / len(val_dataloader),
            "val/loss_cls" : cls_losses / len(val_dataloader),
            "val/loss_total" : total_losses / len(val_dataloader),
            "val_precision" : (cat1_precision + cat2_precision) /2 ,
            "val_recall" : (cat1_recall + cat2_recall) /2 ,
            "val_mAP" :  mAP}
            wandb.log(val_log_dict)
            val_img_dict =  {
                "val/PR_plot": PR_plot,
                "val/R_plot": R_plot,
                "val/P_plot": P_plot,
                "val/F1_plot": F1_plot , 
            }
            wandb.log(val_img_dict)
            
            
            torch.save(model.state_dict(), f'./ckpts/epoch_{epoch}.pth')

            print(f'step : {j}\nloss_bbox : {bbox_losses / len(val_dataloader)}   \
                    loss_cls_total : {cls_losses /len(val_dataloader)}  total_loss : {total_losses / len(val_dataloader)}')
    
    print('-' * 30)

 