
import os
import numpy as np
import shutil
import cv2
import json
import glob
import matplotlib.pyplot as plt
import pandas as pd
import warnings
import random
import argparse
import os.path as osp
import sys
import tqdm
sys.path.append('/home/user304/users/jiwon/defect_detection/')
from utils import type_converter as tcvt

warnings.filterwarnings(action='ignore')
parser = argparse.ArgumentParser(
    description='labelme의 creatML format을 cocoformat으로 바꾸는 script')

parser.add_argument('--label_path')
parser.add_argument('--save_path')

args = parser.parse_args()

img_paths = glob.glob('./dataset/images/*')
label_paths = glob.glob(osp.join(args.label_path, '*'))

# assert osp.splitext(args.label_path)[-1] == 'json'
df_anno = pd.DataFrame(columns=['id', 'category_id', 'bbox', 'image_id'])
df_img = pd.DataFrame(columns=['id', 'file_name', 'RESOLUTION', 'height' , 'width'])

anno_idx = 1
for idx, img_path in tqdm.tqdm(enumerate(img_paths, start=1)):
    # json to img file path
    label_path = img_path.replace('images', 'labels').replace('jpg', 'json')

    # load json file
    img = cv2.imread(img_path, cv2.IMREAD_GRAYSCALE)

    height, width = img.shape
    resolution = int(height * width)
    file_name = img_path

    df_img = df_img.append(
        {'id': idx,
            'file_name': file_name,
         'RESOLUTION': resolution,
         'height': height,
         'width': width}, ignore_index=True)

    if osp.exists(label_path):
        with open(label_path, 'r') as file:
            json_file = json.load(file)
        # get annotations
        annos = json_file[0]['annotations']
        for anno in annos:
            category = anno['label']
            bbox = {k: int(v) for k, v in anno['coordinates'].items()}
            bbox = list(bbox.values())
            df_anno = df_anno.append(
                {'id': anno_idx,
                'category_id': category,
                'bbox': bbox,
                'image_id': idx}, ignore_index=True)
            anno_idx += 1

# file path에서 dataset경로 제거
df_img['file_name'] = df_img['file_name'].apply(lambda x: x.replace('./dataset/', ''))

df_anno['bbox'] = df_anno['bbox'].apply( lambda x: tcvt.createML_to_coco(x, dim=1))
df_anno['area'] = df_anno['bbox'].apply(lambda x: x[2] * x[3])

category_to_id = {'over': 1,'under': 2}

df_anno['category_id'] = df_anno['category_id'].apply(lambda x: category_to_id[x])
df_anno['iscrowd'] = 0

print('samples_num :', df_img.values.shape[0])
# cvt COCO
df_coco = {}
df_coco['images'] = df_img.to_dict('records')
df_coco['annotations'] = df_anno.to_dict('records')
df_coco['categories'] = [{'id': category_to_id['over'], 'name': 'over', 'supercategory': 'defect'},
                         {'id': category_to_id['under'],'name': 'under', 'supercategory': 'defect'}]

with open(args.save_path, "w") as json_file:
    json.dump(df_coco, json_file, indent=4)
