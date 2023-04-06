from torch.utils.data import DataLoader
import torch
import torch.utils.data as data
from pycocotools.coco import COCO
import os
import cv2
import numpy as np
import torch
from torch.utils.data import Sampler
from typing import Iterator, Iterable, Optional, Sequence, List, TypeVar, Generic, Sized, Union
import matplotlib.pyplot as plt
import math
import random
import albumentations as A
from albumentations.pytorch import ToTensorV2
import copy


def coco_cv2(bboxes):
    return [[x, y, x+w, y+h] for x, y, w, h in bboxes]


class CustomDataset(data.Dataset):
    def __init__(self, data_dir, transformer=None):
        super(CustomDataset, self).__init__()
        self.coco = COCO(data_dir)
        self.img_idx_list = self.coco.getImgIds()
        self.transformer = transformer
        ori_shape = (596, 784, 3)
        pad_shape = (512, 512, 3)
        img_shape = (512, 512, 3)
        xy_ratio = np.array(
            list(y/x for x, y in zip(ori_shape, img_shape))[:2], dtype=np.float32)[::-1]
        scale_factor = np.tile(xy_ratio, 2)
        self.img_info = {'filename': './dataset/', 'ori_filename': '', 'ori_shape': ori_shape,
                         'img_shape': img_shape, 'pad_shape': pad_shape, 'scale_factor': scale_factor, 'flip': False,
                         'flip_direction': None,
                         'img_norm_cfg': {'mean': np.array([46.84, 46.84, 46.84], dtype=np.float32),
                                          'std': np.array([48.73, 48.73, 48.73], dtype=np.float32)},
                         'to_rgb': True}

    def __len__(self):
        return len(self.img_idx_list)

    def __getitem__(self, idx):
        img_meta, image = self.get_image(idx)
        bboxes, class_ids = self.get_label(idx)

        if self.transformer:
            transformed_data = self.transformer(
                image=image, bboxes=bboxes, class_ids=class_ids)
            image = transformed_data['image']
            bboxes = np.array(coco_cv2(transformed_data['bboxes']))
            class_ids = np.array(transformed_data['class_ids'])

            if len(class_ids) == 0:
                bboxes = torch.empty(0, 4, dtype=torch.float32)
                class_ids = torch.empty(0, 1, dtype=torch.long)

        return img_meta, image, bboxes, class_ids

    def get_image(self, idx):
        img_meta = copy.deepcopy(self.img_info)
        img_info = self.coco.loadImgs(ids=idx)[0]
        filename = img_info['file_name']
        image_path = os.path.join('./dataset', filename)
        image = cv2.imread(image_path)
        image = cv2.cvtColor(image, cv2.COLOR_BGR2RGB)
        img_meta['filename'] += filename
        img_meta['ori_filename'] = filename
        return img_meta, image

    def get_label(self, idx):
        ann_ids = self.coco.getAnnIds(imgIds=idx)
        anns = self.coco.loadAnns(ann_ids)
        bboxes = np.array([ann['bbox'] for ann in anns])
        class_ids = np.array([ann['category_id'] for ann in anns])

        return bboxes, class_ids


class CustomRandomSampler(Sampler[int]):
    data_source: Sized
    replacement: bool

    def __init__(self, data_source: Sized,
                 num_samples: Optional[int] = None, generator=None) -> None:
        self.data_source = data_source
        self._num_samples = num_samples
        self.generator = generator
        cat1_imgs = set(data_source.coco.getImgIds(
            catIds=data_source.coco.getCatIds()[0]))
        cat2_imgs = set(data_source.coco.getImgIds(
            catIds=data_source.coco.getCatIds()[1]))
        self.defect_imgs = list(cat2_imgs | cat1_imgs)
        self.other_imgs = list(
            set(data_source.coco.getImgIds()) - set(cat2_imgs | cat1_imgs))

    @property
    def num_samples(self) -> int:
        # dataset size might change at runtime
        if self._num_samples is None:
            return len(self.data_source)
        return self._num_samples

    def __iter__(self) -> Iterator[int]:
        n_n = len(self.other_imgs)
        n_d = len(self.defect_imgs)
        if self.generator is None:
            seed = int(torch.empty((), dtype=torch.int64).random_().item())
            generator = torch.Generator()
            generator.manual_seed(seed)
        else:
            generator = self.generator

        if self.dtype == 'defect':
            for _ in range(self.num_samples // 32):
                yield from [self.defect_imgs[i] for i in torch.randint(high=n_d, size=(32,), dtype=torch.int64, generator=generator).tolist()]
            yield from [self.defect_imgs[i] for i in torch.randint(high=n_d, size=(self.num_samples % 32,), dtype=torch.int64, generator=generator).tolist()]
        else:
            for _ in range(n_n // n_n):
                yield from [self.other_imgs[i] for i in torch.randperm(n_n, generator=generator).tolist()]
            yield from [self.other_imgs[i] for i in torch.randperm(n_n, generator=generator).tolist()[:n_n % n_n]]

    def __len__(self) -> int:
        if self.dtype == 'defect':
            return len(self.defect_imgs)
        if self.dtype == 'normal':
            return len(self.other_imgs)
        else:
            return len(self.data_source)

    def __call__(self, dtype) -> str:
        self.dtype = dtype
        return self


class CustomBatchSampler(Sampler[List[int]]):
    def __init__(self, sampler_1: Union[Sampler[int], Iterable[int]],
                 sampler_2: Union[Sampler[int], Iterable[int]],
                 batch_size: int, fixed_size: int, drop_last=True) -> None:
        self.sampler_1 = sampler_1
        self.sampler_2 = sampler_2
        self.batch_size = batch_size
        self.fixed_size = fixed_size
        self.drop_last = drop_last

    def __iter__(self) -> Iterator[List[int]]:
        batch = [0] * self.batch_size
        idx_in_batch = 0
        fix_size = 0
        normal_sampler = iter(self.sampler_1('normal'))
        defect_sampler_iter = iter(self.sampler_2('defect'))
        if self.drop_last:
            while True:
                try:
                    idx = next(normal_sampler)
                    while fix_size < self.fixed_size:
                        batch[idx_in_batch] = next(defect_sampler_iter)
                        fix_size += 1
                        idx_in_batch += 1
                    batch[idx_in_batch] = idx
                    idx_in_batch += 1
                    if idx_in_batch == self.batch_size:
                        random.shuffle(batch)
                        yield batch
                        idx_in_batch = 0
                        fix_size = 0
                        batch = [0] * self.batch_size
                except StopIteration:
                    break

    def __len__(self) -> int:
        num_steps = math.ceil(
            len(self.sampler_2.other_imgs) / (self.batch_size - self.fixed_size))
        return num_steps


class CustomDataLoader(data.DataLoader):
    def __init__(self, *args, **kwargs):
        super(CustomDataLoader, self).__init__(*args, **kwargs)
        self.collate_fn = collate_fn


def collate_fn(batch):
    image_list = []
    bboxes_list = []
    cls_ids_list = []
    img_metas = []
    for img_meta, image, bboxes, class_ids in batch:
        image_list.append(image)
        img_metas.append(img_meta)
        if isinstance(bboxes, (np.ndarray, np.generic)):
            bboxes_list.append(torch.tensor(bboxes, dtype=torch.float32))
            cls_ids_list.append(torch.tensor(class_ids, dtype=torch.long))
        else:
            bboxes_list.append(bboxes)
            cls_ids_list.append(class_ids)

    return img_metas, torch.stack(image_list, dim=0), bboxes_list, cls_ids_list


def import_data_loader(data_dir, transformer):
    dataset = CustomDataset(data_dir=data_dir, transformer=transformer)
    random_sampler_1 = CustomRandomSampler(dataset)
    random_sampler_2 = CustomRandomSampler(dataset)
    batchsampler = CustomBatchSampler(
        random_sampler_1, random_sampler_2, batch_size=8, fixed_size=2)
    dataloader = CustomDataLoader(
        dataset, batch_sampler=batchsampler, num_workers=8, collate_fn=collate_fn)

    return dataloader


class TestCustomDataset(data.Dataset):
    def __init__(self, data_dir, transformer=None):
        super(TestCustomDataset, self).__init__()
        self.coco = COCO(data_dir)
        self.img_idx_list = self.coco.getImgIds()
        self.transformer = transformer
        ori_shape = (596, 784, 3)
        pad_shape = (512, 512, 3)
        img_shape = (512, 512, 3)
        xy_ratio = np.array(
            list(y/x for x, y in zip(ori_shape, img_shape))[:2], dtype=np.float32)[::-1]
        scale_factor = np.tile(xy_ratio, 2)
        self.img_info = {'filename': './dataset/', 'ori_filename': '', 'ori_shape': ori_shape,
                         'img_shape': img_shape, 'pad_shape': pad_shape, 'scale_factor': scale_factor, 'flip': False,
                         'flip_direction': None,
                         'img_norm_cfg': {'mean': np.array([46.84, 46.84, 46.84], dtype=np.float32),
                                          'std': np.array([48.73, 48.73, 48.73], dtype=np.float32)},
                         'to_rgb': True}

    def __len__(self):
        return len(self.img_idx_list)

    def __getitem__(self, idx):
        img_meta, image = self.get_image(idx)
        bboxes, class_ids = self.get_label(idx)

        if self.transformer:
            transformed_data = self.transformer(
                image=image, bboxes=bboxes, class_ids=class_ids)
            image = transformed_data['image']
            bboxes = np.array(coco_cv2(transformed_data['bboxes']))
            class_ids = np.array(transformed_data['class_ids'])

            if len(class_ids) == 0:
                bboxes = torch.empty(0, 4, dtype=torch.float32)
                class_ids = torch.empty(0, 1, dtype=torch.long)

        return img_meta, image, bboxes, class_ids

    def get_image(self, idx):
        idx = self.img_idx_list[idx]
        img_meta = copy.deepcopy(self.img_info)
        img_info = self.coco.loadImgs(ids=idx)[0]
        filename = img_info['file_name']
        image_path = os.path.join('./dataset', filename)
        image = cv2.imread(image_path)
        image = cv2.cvtColor(image, cv2.COLOR_BGR2RGB)
        img_meta['filename'] += filename
        img_meta['ori_filename'] = filename
        return img_meta, image

    def get_label(self, idx):
        idx = self.img_idx_list[idx]
        ann_ids = self.coco.getAnnIds(imgIds=idx)
        anns = self.coco.loadAnns(ann_ids)
        bboxes = np.array([ann['bbox'] for ann in anns])
        class_ids = np.array([ann['category_id'] for ann in anns])

        return bboxes, class_ids


# if __name__ == "__main__":
#     transformer = A.Compose([
#         A.Resize(512, 512),
#         A.OneOf([
#             A.Rotate(limit=(0, 90), p=0.7),
#             A.Compose([
#                 A.HorizontalFlip(always_apply=True),
#                 A.Rotate(limit=(-90, 0), p=0.7)
#             ])
#         ], p=0.7),
#         A.OneOf([
#             A.RandomBrightnessContrast(brightness_limit=(
#                 0, 0.2), contrast_limit=(0, 0.3), p=0.7),
#             A.CLAHE(clip_limit=4.0, tile_grid_size=(8, 8), p=0.7),
#             A.Sharpen(alpha=(0.2, 0.8), lightness=(0.5, 1.0), p=0.7)
#         ], p=0.7),
#         A.Normalize(mean=(0.1837, 0.1837, 0.1837),
#                     std=(0.1911, 0.1911, 0.1911)),
#         ToTensorV2()
#     ], bbox_params=A.BboxParams(format='coco', label_fields=['class_ids']))

#     data_dir = "../info/train.json"
#     dataset = CustomDataset(data_dir=data_dir, transformer=transformer)
#     random_sampler_1 = CustomRandomSampler(dataset)
#     random_sampler_2 = CustomRandomSampler(dataset)
#     batchsampler = CustomBatchSampler(
#         random_sampler_1, random_sampler_2, batch_size=8, fixed_size=3)
#     dataloader = CustomDataLoader(
#         dataset, batch_sampler=batchsampler, num_workers=8, collate_fn=collate_fn)

#     for i, (imgs, bboxes, labels) in enumerate(dataloader):
#         pass

#     print('step :', i)
