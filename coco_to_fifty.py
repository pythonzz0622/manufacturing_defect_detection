import fiftyone.brain as fob
import json
import fiftyone as fo
from pycocotools.coco import COCO
import os
import re
from tqdm import tqdm
# get groundtruth info
coco = COCO('./dataset/test.json')
preds_path = './result.json'

if preds_path: 
    with open(preds_path, 'r') as f:
        pred_list = json.load(f)
else:
    pred_list = None

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
                         bounding_box=bbox, confidence=pred['score'] , area = int(pred['bbox'][2] * pred['bbox'][3]))
        )
    return prediction_samples


img_path_to_id = {re.sub('.jpg', '', os.path.basename(
    x['file_name'])): x['id'] for x in coco.loadImgs(coco.getImgIds())}
# id : category 매칭
id_to_cat = {v['id']: v['name'] for v in coco.loadCats(coco.getCatIds())}


# img 별 iteration
samples = []
for img_id in tqdm(coco.getImgIds()):
    # fiftyone sample에 img 삽입
    sample = fo.Sample(filepath='./dataset/' +
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


    # ground truth가 있을시에 sample 삽입
    if detections:
        sample['ground_truth'] = fo.Detections(detections=detections)

    if pred_list:
        preds = get_pred_samples(pred_list, img_id=img_id) 
        sample['preds'] = fo.Detections(detections=preds)
        
    samples.append(sample)


# fob.compute_similarity(dataset, brain_key="similarity", num_workers=64)
# dataset title 넣기
dataset = fo.Dataset()
dataset.add_samples(samples)
# dataset.add_sample_field( field_name='preds.ground_truth.area', ftype = fo.core.fields.IntField , description ='An area')
if pred_list:
    dataset.add_sample_field( field_name='preds.detections.area', ftype = fo.core.fields.IntField , description ='An area')
# fob.compute_uniqueness(dataset, num_workers=64)
# fob.compute_mistakenness(samples=dataset, pred_field='preds', label_field='ground_truth')
dataset.save()

if __name__ == "__main__":
    session = fo.launch_app(dataset, port=8888, address="0.0.0.0")
    session.wait()