import json
import fiftyone as fo
import glob
from pycocotools.coco import COCO

# img_list = glob.glob('../dataset/defect/*') +  glob.glob('../dataset/normal/*')
# dataset = fo.Dataset.from_images(img_list)

coco = COCO('../info/test.json')
img_list = ['../dataset/' + anno['file_name']
            for anno in coco.loadImgs(coco.getImgIds())]
# Create a dataset from a glob pattern of images
dataset = fo.Dataset.from_images(img_list)

# if __name__=="__main__":
#     dataset = fo.Dataset.from_images(img_list)
#     session = fo.launch_app(dataset , port =5000 , address= "0.0.0.0" )
#     session.wait()
img_id = coco.getImgIds()[11]
coco.getAnnIds(img_id)
coco.loadCats(coco.getCatIds())

id_to_cat = {v['id']: v['name'] for v in coco.loadCats(coco.getCatIds())}

with open('../mmdetection/result/retinanet_swin-l/result.bbox.json', 'r') as f:
    data = json.load(f)


def min_max(bbox):
    x, w = [x / 784 for x in bbox[::2]]
    y, h = [x / 596 for x in bbox[1::2]]
    return [x, y, w, h]


samples = []
for img_id in coco.getImgIds():
    sample = fo.Sample(filepath='../dataset/' +
                       coco.loadImgs(img_id)[0]['file_name'])

    detections = []
    for ann_id in coco.getAnnIds(imgIds=img_id):
        obj = coco.loadAnns(ann_id)[0]
        label = id_to_cat[obj['category_id']]
        bounding_box = min_max(obj['bbox'])

        detections.append(
            fo.Detection(label=label, bounding_box=bounding_box)
        )

    predictions_anno = [
        data_i for data_i in data if data_i['image_id'] == img_id]
    predictions = []
    for pred in predictions_anno:
        bbox = min_max(pred['bbox'])
        predictions.append(
            fo.Detection(label=id_to_cat[pred['category_id']],
                         bounding_box=bbox, confidence=pred['score'])
        )

    if detections:
        sample['ground_truth'] = fo.Detections(detections=detections)
        samples.append(sample)

    if predictions:
        sample['prediction'] = fo.Detections(detections=predictions)
        samples.append(sample)

dataset = fo.Dataset('my-detection-dataset')
dataset.add_samples(samples)


if __name__ == "__main__":
    session = fo.launch_app(dataset, port=8842, address="0.0.0.0")
    session.wait()
