import numpy as np
from pycocotools.coco import COCO
import os
import albumentations as A
import matplotlib.pyplot as plt
import cv2
from ipywidgets import interact
import math
A.__version__


class Img_visualization():
    '''
    albumentation uses for values [x1 ,y1 , x2 ,y2]
    '''

    def __init__(self) -> None:
        pass

    def coco_to_cv2(self, bbox_list):
        cv2_bbox_list = []
        for bbox in bbox_list:
            xmin, ymin, width, height = bbox
            cv2_bbox_list.append([xmin, ymin, xmin + width, ymin + height])
        return cv2_bbox_list

    def trans(self, img, transformer):
        transform = transformer
        augmented_image = transform(image=img)['image']
        label = str(transformer).split('(')[0]
        return augmented_image, label

    def trans_bbox(self, img, bbox_list, transformer):
        transform = transformer
        trans_img = transform(image=img, bboxes=bbox_list)
        augmented_image = trans_img['image']
        augmented_bboxes = trans_img['bboxes']
        label = str(transformer).split('(')[0]
        return augmented_image, augmented_bboxes, label

    def pixel_level_aug(self, img, bbox_list, save_path):
        pixel_level = ['Superpixels', 'GlassBlur', 'RandomFog', 'ISONoise', 'UnsharpMask', 'RandomShadow', 'Posterize', 'Normalize', 'Sharpen',
                       'Equalize', 'Blur', 'MotionBlur', 'CLAHE', 'RandomBrightnessContrast', 'RandomToneCurve', 'MultiplicativeNoise', 'Solarize', 'GaussNoise',
                       'HueSaturationValue', 'Downscale', 'RGBShift', 'RingingOvershoot', 'GaussianBlur', 'ToSepia', 'FromFloat', 'ImageCompression', 'ChannelDropout',
                       'RandomSnow', 'AdvancedBlur', 'RandomSunFlare', 'ColorJitter', 'RandomRain', 'ChannelShuffle', 'MedianBlur', 'InvertImg', 'ToGray',
                       'ToFloat', 'RandomGamma', 'FancyPCA', 'Emboss']
        other_pixel = ['FDA', 'HistogramMatching',
                       'PixelDistributionAdaptation', 'bboxlateTransform']

        pixel_level = list(set(pixel_level) - set(other_pixel))

        col, row = math.ceil(len(pixel_level) / 5), 5
        fig, ax = plt.subplots(col, row, figsize=(20, 35))
        c = 0
        x = 1
        for i in range(8):
            for j in range(5):
                transformer = getattr(A, pixel_level[c])(always_apply=True)
                aug_img, label = self.trans(img, transformer)
                ax[i, j].set_title(label)
                ax[i, j].axis('off')
                if bbox_list:
                    for bbox in bbox_list:
                        cv2.rectangle(aug_img, (int(bbox[0]),  int(bbox[1])), (int(
                            bbox[2]), int(bbox[3])), (255, 255, 0), 1)
                ax[i, j].imshow(aug_img)
                c += 1
        plt.tight_layout()
        if save_path:
            plt.savefig(save_path, dpi=300)
        plt.show()

    def spatial_level_aug(self, img, bbox_list, save_path):
        spatial_level = ['GridDropout', 'Rotate', 'Transpose', 'LongestMaxSize', 'GridDistortion', 'SafeRotate', 'RandomScale', 'ElasticTransform', 'SmallestMaxSize',
                         'PiecewiseAffine', 'ShiftScaleRotate', 'PixelDropout', 'Perspective', 'CoarseDropout', 'RandomGridShuffle', 'Flip', 'PadIfNeeded', 'VerticalFlip',
                         'NoOp', 'HorizontalFlip', 'OpticalDistortion', 'Affine', 'RandomRotate90', 'Lambda']
        other_level = ['CenterCrop', 'Crop', 'CropAndPad', 'CropNonEmptyMaskIfExists', 'MaskDropout', 'RandomCrop', 'RandomCropNearBBox',
                       'RandomResizedCrop', 'RandomSizedBBoxSafeCrop', 'RandomSizedCrop', 'Resize']

        spatial_level = list(set(spatial_level) - set(other_level))

        aug_list = [
            A.CenterCrop(height=100, width=100, always_apply=True),
            A.Crop(x_min=0, y_min=0, x_max=100, y_max=100, always_apply=True),
            A.CropAndPad(px=10, always_apply=True),
            A.RandomResizedCrop(height=100, width=100, always_apply=True),
            A.RandomSizedCrop(min_max_height=(20, 100), height=100,
                              width=100, always_apply=True)
        ]

        aug_bbox = [
            A.RandomSizedBBoxSafeCrop(
                height=300, width=300, always_apply=True),
            A.RandomCropNearBBox(always_apply=True)
        ]

        col, row = math.ceil(len(spatial_level) / 5), 5
        fig, ax = plt.subplots(col, row, figsize=(15, 15))
        c = 0
        for i in range(5):
            for j in range(5):
                try:
                    transformer = getattr(
                        A, spatial_level[c])(always_apply=True)
                    aug_img, aug_bbox, label = self.trans_bbox(
                        img, bbox_list, transformer)
                    ax[i, j].set_title(label)
                    ax[i, j].axis('off')
                    ax[i, j].imshow(aug_img)
                    c += 1
                except IndexError:
                    pass
        if save_path:
            plt.savefig(save_path, dpi=300)
        plt.show()

        fig, ax = plt.subplots(1, 5, figsize=(15, 3))
        c = 0
        for i in range(5):
            transformer = aug_list[i]
            aug_img, aug_bbox, label = self.trans_bbox(
                img, bbox_list, transformer)
            ax[i].set_title(label)
            ax[i].axis('off')
            ax[i].imshow(aug_img)
        plt.show()

        # for i in range(len(aug_bbox)):
        #     aug = aug_bbox[i](image=img, bboxes=bbox_list)
        #     fig, ax = plt.subplots(1, 4, figsize=(12, 3))
        #     fig.suptitle(str(aug_bbox[i]).split('(')[0])
        #     ax[0].imshow(img)
        #     IMG_BBOX = img.copy()
        #     for bbox in bbox_list:
        #         cv2.rectangle(IMG_BBOX, (int(bbox[0]),  int(bbox[1])), (int(bbox[2]), int(bbox[3])),
        #                       (255, 255, 0), 1)
        #     ax[1].imshow(IMG_BBOX)
        #     ax[2].imshow(aug['image'])
        #     for bbox in aug['bboxes']:
        #         cv2.rectangle(IMG_BBOX, (int(bbox[0]),  int(bbox[1])), (int(bbox[2]), int(bbox[3])),
        #                       (255, 255, 0), 1)
        #     ax[3].imshow(IMG_BBOX)
        #     plt.show()
