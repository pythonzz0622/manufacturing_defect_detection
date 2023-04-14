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
import utils.type_converter  as tcvt
import json
from tqdm import tqdm
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
# run = wandb.init(
#   project="retina-fpn-swin-l",
#   notes="test_experiment",
#   entity = "gnu-ml-lab",
#   tags=["baseline"]
# )

# # wandb.define_metric("train/step")
# wandb.define_metric("val/epoch")
# # wandb.define_metric("train/*", step_metric="train/step")
# wandb.define_metric("val/*", step_metric="val/epoch")
device = torch.device('cuda')
val_dataset = customLoader.TestCustomDataset(
    './dataset/coco.json', transformer=val_transformer)
val_dataloader = DataLoader(dataset=val_dataset, batch_size=8,
                            collate_fn=customLoader.collate_fn)
cvt = tcvt.COCO_converter('./dataset/coco.json')

de_std = tuple(std * 255 for std in std)
de_mean = tuple(mean * 255 for mean in mean)

model = retina_fpn_swin_l()

optimizer = optim.AdamW(model.parameters(),
                        lr=0.0001 / 8,
                        betas=(0.9, 0.999),
                        weight_decay=0.05,
                        )

model = model.cuda()
model.load_state_dict(torch.load('./ckpts/rfs-renew_data/epoch_3.pth'))

cls_losses = 0
bbox_losses = 0
total_losses = 0

preds_list = []
with torch.no_grad():
    model.eval()
    for j, (img_metas, images, bboxes, labels) in tqdm(enumerate(val_dataloader)):

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
                'category_id' : pred_list[1][1][i]} 
                for pred_list in preds_list  
                for i in range(len(pred_list[1][1]) ) if tcvt.get_area(pred_list[1][0][i][:4]) > 0 ]
        

    with open('result.json', "w") as json_file:
        json.dump(result, json_file, indent=4 , cls = tcvt.NpEncoder)
