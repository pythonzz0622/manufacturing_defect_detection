import json
import fiftyone as fo
from pycocotools.coco import COCO
import os
import re

# image 정규화


def min_max(bbox):
    x, w = [x / 784 for x in bbox[::2]]
    y, h = [x / 596 for x in bbox[1::2]]
    return [x, y, w, h]

# coco data로 fiftyone에 sample 삽입


def get_pred_samples(json_data, img_id):
    # image에 해당하는 annotations
    pred_annos = [data for data in json_data if data['image_id'] == img_id]
    prediction_samples = []
    for pred in pred_annos:
        bbox = min_max(pred['bbox'])
        prediction_samples.append(
            fo.Detection(label=id_to_cat[pred['category_id']],
                         bounding_box=bbox, confidence=pred['score'])
        )
    return prediction_samples


# get groundtruth info
coco = COCO('../info/test.json')

img_path_to_id = {re.sub('.jpg', '', os.path.basename(
    x['file_name'])): x['id'] for x in coco.loadImgs(coco.getImgIds())}

img_path_list = ['../dataset/' + anno['file_name']
                 for anno in coco.loadImgs(coco.getImgIds())]

# Create a dataset from a glob pattern of images
dataset = fo.Dataset.from_images(img_path_list)

# id : category 매칭
id_to_cat = {v['id']: v['name'] for v in coco.loadCats(coco.getCatIds())}

# load coo file
with open('../mmdetection/result/retinanet_swin-l/result.bbox.json', 'r') as f:
    swin_data = json.load(f)

# load coco file
with open('../mmdetection/result/faster_rcnn/result.bbox.json', 'r') as f:
    rcnn_data = json.load(f)

# load yolo file
with open('../yolov7/runs/test/yolov7_640_val2/last_predictions.json', 'r') as f:
    yolo_data = json.load(f)

# yolo file coco format으로 맞추기
for i in range(len(yolo_data)):
    yolo_data[i]['image_id'] = img_path_to_id[yolo_data[i]['image_id']]
    yolo_data[i]['category_id'] += 1

# img 별 iteration
samples = []
for img_id in coco.getImgIds():
    # fiftyone sample에 img 삽입
    sample = fo.Sample(filepath='../dataset/' +
                       coco.loadImgs(img_id)[0]['file_name'])

    detections = []
    # annotation 별 iteration
    for ann_id in coco.getAnnIds(imgIds=img_id):
        obj = coco.loadAnns(ann_id)[0]
        label = id_to_cat[obj['category_id']]
        bbox = min_max(obj['bbox'])

        # fiftyone detections sample에 annotation 삽입
        detections.append(
            fo.Detection(label=label, bounding_box=bbox)
        )

    # model 별로 fiftyone annotation 표본 얻기
    swin_preds = get_pred_samples(swin_data, img_id=img_id)
    rcnn_preds = get_pred_samples(rcnn_data, img_id=img_id)
    yolo_preds = get_pred_samples(yolo_data, img_id=img_id)

    # ground truth가 있을시에 sample 삽입
    if detections:
        sample['ground_truth'] = fo.Detections(detections=detections)

    # model 별로 annotation 표본 삽입
    sample['swin-transformer'] = fo.Detections(detections=swin_preds)
    sample['faster-rcnn'] = fo.Detections(detections=rcnn_preds)
    sample['yolov7'] = fo.Detections(detections=yolo_preds)

    samples.append(sample)

# dataset title 넣기
dataset = fo.Dataset('my-detection-dataset')
dataset.add_samples(samples)
# meta data 계산
dataset.compute_metadata(num_workers=10)


if __name__ == "__main__":
    session = fo.launch_app(dataset, port=8842, address="0.0.0.0")
    session.wait()
