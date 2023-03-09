import fiftyone.brain as fob
import json
import fiftyone as fo
from pycocotools.coco import COCO
import os
import re

# get groundtruth info
coco = COCO('../info/all.json')


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


img_path_to_id = {re.sub('.jpg', '', os.path.basename(
    x['file_name'])): x['id'] for x in coco.loadImgs(coco.getImgIds())}

img_path_list = ['../dataset/' + anno['file_name']
                 for anno in coco.loadImgs(coco.getImgIds())]

# Create a dataset from a glob pattern of images
dataset = fo.Dataset.from_images(img_path_list)

# id : category 매칭
id_to_cat = {v['id']: v['name'] for v in coco.loadCats(coco.getCatIds())}

# load coo file
with open('../mmdetection/result/new_retinanet_swin-l/result.bbox.json', 'r') as f:
    swin_data = json.load(f)

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

    # ground truth가 있을시에 sample 삽입
    if detections:
        sample['ground_truth'] = fo.Detections(detections=detections)

    # model 별로 annotation 표본 삽입
    sample['swin-transformer'] = fo.Detections(detections=swin_preds)

    samples.append(sample)

# dataset title 넣기
dataset = fo.Dataset()
dataset.add_samples(samples)
# meta data 계산
# dataset.compute_metadata(num_workers=64)
# fob.compute_uniqueness(dataset, num_workers=64)
# fob.compute_similarity(dataset, brain_key="similarity", num_workers=64)
fob.compute_mistakenness(
    samples=dataset, pred_field='swin-transformer', label_field='ground_truth')

dataset.evaluate_detections('swin-transformer', gt_field='ground_truth',
                            eval_key='eval', classes=None, missing=None, method=None, iou=0.5,
                            use_masks=False, use_boxes=False, classwise=True, dynamic=True)


if __name__ == "__main__":
    session = fo.launch_app(dataset, port=5000, address="0.0.0.0")
    session.wait()
