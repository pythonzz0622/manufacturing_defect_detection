import glob
from pycocotools.coco import COCO
import os


# os.makedirs('../info/yolov7/labels/train', exist_ok=True)
# os.makedirs('../info/yolov7/labels/test', exist_ok=True)
# os.makedirs('../info/yolov7/labels/all', exist_ok=True)
# os.makedirs('../dataset/')


def _coco_to_yolo(cls_id, bbox):
    '''
    coco format : [x_1 , y_1 , w , h] 인 반면에 
    yolo format : [x_c , y_c , w , h] {annotation | 0 < anntation <= 1} 이므로 변환과정이 필요함
    '''
    # normalization
    x, w = [x / 784 for x in bbox[::2]]
    y, h = [x / 596 for x in bbox[1::2]]

    # x1 , y1 -> x_c , y_c
    x_c = x + (w / 2)
    y_c = y + (h / 2)

    return [cls_id - 1, x_c, y_c, w, h]


def mk_txt(d_type, data_path):
    '''
    coco format : json file 내 image 정보, label 정보가 모두 들어가있음
    yolo format : txt파일로 이뤄져있으며 image 경로를 저장하는 txt파일 , label파일이 image별로 다 따로 들어가있음
    ex) ├── labels
        │     ├── 001.txt {category_id , x_c , y_c , w ,h  ...}
        │     ├── 002.txt
        │          ...
        │   
        ├── test.txt  {001.jpg , 002.jpg ... }
        └── train.txt {103.jpg  , 105.jpg  ...}
    '''
    coco = COCO(data_path)
    img_id_list = coco.getImgIds()

    # get img path list
    img_list = ['../dataset/' + x['file_name']
                for x in coco.loadImgs(img_id_list)]

    # write image_path_list
    with open(f'../info/yolov7/{d_type}.txt', 'w') as f:
        for img_path in img_list:
            f.write(img_path + '\n')

    # get img_name
    for img_id in img_id_list:
        img_path = coco.loadImgs(ids=img_id)[0]['file_name']
        label_name = os.path.basename(img_path).replace('jpg', 'txt')

        anns_ids = coco.getAnnIds(imgIds=img_id)
        # save anno info
        with open(f'../info/yolov7/labels/{label_name}', 'w') as f:
            for ann_id in anns_ids:
                obj = coco.loadAnns(ann_id)[0]
                label_txt = ' '.join(
                    [str(x) for x in _coco_to_yolo(obj['category_id'], obj['bbox'])])
                f.write(label_txt + '\n')


if __name__ == '__main__':
    mk_txt(d_type='train', data_path='../info/train.json')
    mk_txt(d_type='test', data_path='../info/test.json')
