from custom_dataloader import import_data_loader
import custom_dataloader
from custom_model import retina_fpn_swin_l
import albumentations as A
from albumentations.pytorch import ToTensorV2
import torch.optim as optim
import torch.nn as nn
from torchvision.ops import focal_loss
import torch
import matplotlib.pyplot as plt
import numpy as np
import cv2
from torch.utils.data import DataLoader

###################################### transformer ##############################
mean = (0.485, 0.456, 0.406)
std = (0.229, 0.224, 0.225)
transformer = A.Compose([
    A.Resize(512, 512),
    A.Flip(),
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
device = torch.device('cuda')

train_dataset = custom_dataloader.CustomDataset(
    '../info/train.json', transformer=transformer)
train_dataloader = import_data_loader(
    '../info/train.json', transformer=transformer)
val_dataset = custom_dataloader.CustomDataset(
    '../info/test.json', transformer=val_transformer)
val_dataloader = DataLoader(dataset=val_dataset, batch_size=8,
                            collate_fn=custom_dataloader.collate_fn)


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


epochs = 150
cls_losses = 0
bbox_losses = 0
total_losses = 0
for epoch in range(1, epochs + 1):
    for i, (img_metas, images, bboxes, labels) in enumerate(train_dataloader):
        images = images.cuda()
        model.train()

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
        total_loss.backward()
        optimizer.step()
        optimizer.zero_grad()

        cls_losses += loss_cls_total
        bbox_losses += loss_bbox_total
        total_losses += total_loss
        if i % 100 == 0 and i > 0:
            print(f'[{i} / {len(train_dataloader)}]')
            print(
                f'step : {i}\nloss_bbox : {bbox_losses / 100}  loss_cls_total : {cls_losses /100}  total_loss : {total_losses / 100}')
            cls_losses = 0
            bbox_losses = 0
            total_losses = 0
    # if epoch % 10 == 0:
    #     cls_losses = 0
    #     bbox_losses = 0
    #     total_losses = 0
    #     with torch.no_grad():
    #         model.eval()
    #         for j, (img_metas, images, bboxes, labels) in enumerate(val_dataloader):
    #             bboxes = [bbox.cuda() for bbox in bboxes]
    #             labels = [label.cuda() for label in labels]
    #             outputs = model(images)
    #             losses = model.bbox_head.loss(
    #                 cls_scores=outputs[0],
    #                 bbox_preds=outputs[1],
    #                 gt_bboxes=bboxes,
    #                 gt_labels=labels,
    #                 img_metas=img_metas
    #             )
    #             loss_cls_total = losses['loss_cls'][0] + \
    #                 losses['loss_cls'][1] + losses['loss_cls'][2]
    #             loss_bbox_total = losses['loss_bbox'][0] + \
    #                 losses['loss_bbox'][1] + losses['loss_bbox'][2]
    #             total_loss = loss_cls_total + loss_bbox_total
    #             cls_losses += loss_cls_total
    #             bbox_losses += loss_bbox_total
    #             total_losses += total_loss

    #             bbox_preds = model.bbox_head.get_bboxes(
    #                 cls_scores=outputs[0],
    #                 bbox_preds=outputs[1],
    #                 img_metas=img_metas
    #             )
    #         print(f'step : {i}\nloss_bbox : {bbox_losses / len(val_dataloader)}   \
    #                 loss_cls_total : {cls_losses /len(val_dataloader)}  total_loss : {total_losses / len(val_dataloader)}')
    # print(f'epoch : {epoch}')
    # print('-' * 30)
torch.save(model.state_dict(), './test4.pth')
