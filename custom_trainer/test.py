from custom_dataloader import import_data_loader
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
    A.OneOf([
        A.RandomBrightnessContrast(brightness_limit=(
            0, 0.2), contrast_limit=(0, 0.3), p=0.7),
        A.CLAHE(clip_limit=4.0, tile_grid_size=(8, 8), p=0.7),
        A.Sharpen(alpha=(0.2, 0.8), lightness=(0.5, 1.0), p=0.7)
    ], p=0.7),
    A.Normalize(mean=mean, std=std,
                max_pixel_value=255.0),
    ToTensorV2()
], bbox_params=A.BboxParams(format='coco', label_fields=['class_ids']))
dataloader = import_data_loader('../info/train.json', transformer=transformer)
model = retina_fpn_swin_l()
inputs = next(iter(dataloader))
img_metas, images, bboxes, labels = inputs
bboxes
de_std = tuple(std * 255 for std in std)
de_mean = tuple(mean * 255 for mean in mean)
image = images[1].permute(1, 2, 0)
img = ((image * torch.tensor(de_std)) + torch.tensor(de_mean)).int()
img = img.numpy()
optimizer = optim.AdamW(model.parameters(),
                        lr=0.0001 / 8,
                        betas=(0.9, 0.999),
                        weight_decay=0.05,
                        )
model = model.cuda()
# model.load_state_dict(torch.load('./test.pth'))
epochs = 50
cls_losses = 0
bbox_losses = 0
total_losses = 0
for epoch in range(epochs):
    for i, (img_metas, images, bboxes, labels) in enumerate(dataloader):
        images = images.cuda()
        model.train()

        bboxes = [bbox.cuda() for bbox in bboxes]
        labels = [label.type(torch.long).cuda() for label in labels]
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
            print(f'[{i} / {len(dataloader)}]')
            print(
                f'step : {i}\nloss_bbox : {bbox_losses / 100}  loss_cls_total : {cls_losses /100}  total_loss : {total_losses / 100}')
            cls_losses = 0
            bbox_losses = 0
            total_losses = 0

    print(f'epoch : {epoch}')
    print('-' * 30)
torch.save(model.state_dict(), './test3.pth')
