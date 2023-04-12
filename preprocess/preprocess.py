import os
import os.path as osp
import glob
import shutil

# project path
img_paths = glob.glob('./original_dataset/sd_image_data/**/CAM1/*.jpg')
label_paths = glob.glob('./original_dataset/sd_image_data/**/label/*.json')

print('image 개수 :', len(img_paths))
print('label 개수 :', len(label_paths))

img_dst_path = './dataset/images/'
label_dst_path = './dataset/label/'


for img_path in img_paths:
    dst_path = osp.join(img_dst_path, osp.basename(img_path))
    shutil.move(img_path, dst_path)

for label_path in label_paths:
    dst_path = osp.join(label_dst_path, osp.basename(label_path))
    shutil.move(label_path, dst_path)

# image 개수 31193 , label 개수 624