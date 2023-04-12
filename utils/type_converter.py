import json
import numpy as np
from pycocotools.coco import COCO
import cv2
from funcy import pluck
import matplotlib.pyplot as plt
import math
import os
from pycocotools.cocoeval import COCOeval
import pandas as pd
from datetime import timedelta

class NpEncoder(json.JSONEncoder):
    '''
    numpy to json format
    '''

    def default(self, obj):
        if isinstance(obj, np.integer):
            return int(obj)
        if isinstance(obj, np.floating):
            return float(obj)
        if isinstance(obj, np.ndarray):
            return obj.tolist()
        return super(NpEncoder, self).default(obj)

def sec_to_format(td):
    td = timedelta(seconds=round((td) * 10))
    total_seconds = td.total_seconds()
    hours = total_seconds // 3600
    minutes = (total_seconds % 3600) // 60
    seconds = total_seconds % 60
    print(f"ETA : {int(hours)}:{int(minutes)}:{int(seconds)}")

def createML_to_coco(bboxes, dim=2):
    '''
     createML format : [x_c , y_c , w ,h]
    coco format : [x1 , y1 , w, h]
    '''
    if dim == 1:
        x_c, y_c, w, h = bboxes
        anno = [x_c - (w // 2), y_c - (h // 2),  w, h]
    else:
        anno = [[x_c - (w // 2), y_c - (h // 2),  w, h]
                for x_c, y_c, w, h in bboxes]
    return anno


def coco_to_cv2(bboxes, dim=2):
    # Convert bounding box (x, y, w, h) to bounding box (x1, y1, x2, y2)
    '''
    coco format : [x1,y1, w,h] float or int
    cv2 format : [x1,y1,x2,y2] int
    '''
    if dim == 1:
        x1, y1, w, h = list(map(int, bboxes))
        anno = [x1, y1, x1 + w, y1 + h]
    else:
        anno = [list(map(int, [x1, y1, x1 + w, y1 + h]))
                for x1, y1, w, h in bboxes]
    return anno


def cv2_to_coco(bboxes, dim=2):
    # Convert bounding box (x, y, w, h) to bounding box (x1, y1, x2, y2)
    '''
    cv2 format : [x1,y1,x2,y2] int
    coco format : [x1,y1, w,h] float or int
    '''
    if dim == 1:
        x1, y1, x2, y2 = list(map(int, bboxes))
        anno = [x1, y1, x2-x1, y2-y1]
    else:
        anno = [list(map(int, [x1, y1, x2-x1, y2-y1]))
                for x1, y1, x2, y2 in bboxes]
    return anno

def get_area(bbox , dim=2 , dtype = 'cv2'):
    x1, y1 , x2 ,y2 = bbox
    return int((x2-x1) * (y2-y1))

def _json_parser(json_list, keys=['bbox', 'category_id', 'score']):
    # json에서 원하는 key value값 반환
    result = {}
    for key in keys:
        result[key] = list(pluck(key, json_list))
    return result


class COCO_converter:
    ''' COCO path를 활용해 값을 도출하는 class

    '''

    def __init__(self, coco_gt_path, coco_pred_path=None):
        self.coco_gt = COCO(coco_gt_path)
        if coco_pred_path:
            self.coco_pred = self.coco_gt.loadRes(coco_pred_path)
            self._eval()
        else:
            self.coco_pred = None

        self.img_name_to_id = {os.path.basename(label_info['file_name']): label_info['id']
                               for label_info in self.coco_gt.loadImgs(self.coco_gt.getImgIds())}

        # {imgId : img_name}
        self.imgId_to_name = {v: k for k, v in self.img_name_to_id.items()}

        # {cat_name : catId}
        self.cat_name_to_id = {cat_info['name']: cat_info['id']
                               for cat_info in self.coco_gt.loadCats(self.coco_gt.getCatIds())}

        # {catId : cat_name}
        self.catId_to_name = {v: k for k, v in self.cat_name_to_id.items()}
        self.cat_nums = len(self.coco_gt.getCatIds())

    def __repr__(self):
        expl = ''
        catIds = self.coco_gt.getCatIds()
        catImgs = [self.coco_gt.getImgIds(catIds=catId) for catId in catIds]
        for i, cat in enumerate(catImgs, 1):
            expl += f'{self.catId_to_name[i]} : {cat}\n'
        return expl

    def PR(self, conf_score, catId):
        # confidence score , catId 별 Precision, Recall 반환
        pred_list = []
        truth_list = []
        pred_conf_list = []
        truth_conf_list = []

        # img 별 iteration
        for img_id in sorted(self.coco_gt.getImgIds()):
            result = self.cocoEval.evaluateImg(
                imgId=img_id, catId=catId, aRng=[0, 100000], maxDet=1000)

            if result:
                # 0 idx => IoU thrs 0.5
                pred = result['dtMatches'][0, :]
                pred_conf = result['dtScores']

                pred_list.append(pred.tolist())
                pred_conf_list.append(pred_conf)

                truth = result['gtMatches'][0, :]
                confidence = self._get_conf_list(result)

                truth_list.append(truth.tolist())
                truth_conf_list.append(confidence)

        pred_list = [y for x in pred_list for y in x]
        pred_conf_list = [y for x in pred_conf_list for y in x]

        pred_data = np.array([pred_list, pred_conf_list])
        pred_data = pred_data[:, pred_data[1, :] >= conf_score]
        try:
            precision = np.count_nonzero(
                pred_data[0, :]) / len(pred_data[0, :])
        except ZeroDivisionError:
            precision = 1.0

        truth_list = [y for x in truth_list for y in x]
        truth_conf_list = [y for x in truth_conf_list for y in x]

        truth_data = np.array([truth_list, truth_conf_list])
        truth_data = truth_data[1, :] >= conf_score
        recall = np.count_nonzero(truth_data) / (len(truth_data) + 1e-10)
        return precision, recall

    def img_annos(self, idx):
        ''' 
        Args:
            idx (int) : image coco index
        Returns:
            obj : coco imgIdx의 file_name , ground_truth(optional) , preds(optional) 값 반환
            }
        '''
        # coco_gt에서 이미지 정보 가져오기
        img_info = self.coco_gt.loadImgs(ids=idx)[0]
        img_annos = {'file_name': img_info['file_name']}

        # coco_gt에서 annotation 정보 가져오기
        if self.coco_gt.getAnnIds(imgIds=idx):
            annIds = self.coco_gt.getAnnIds(imgIds=idx)
            ann_infos = self.coco_gt.loadAnns(annIds)
            gt_infos = _json_parser(
                ann_infos, keys=['bbox', 'category_id'])
            img_annos['ground_truth'] = gt_infos

        # coco_pred에서 annotation 정보 가져오기
        if self.coco_pred:
            if self.coco_pred.getAnnIds(imgIds=idx):
                pred_annIds = self.coco_pred.getAnnIds(imgIds=idx)
                ann_infos = self.coco_pred.loadAnns(pred_annIds)
                pred_infos = _json_parser(
                    ann_infos, keys=['bbox', 'category_id', 'score'])
                img_annos['preds'] = pred_infos

        return img_annos

    def _eval(self):
        # coco evaluation 계산
        self.cocoEval = COCOeval(self.coco_gt, self.coco_pred, iouType='bbox')
        self.cocoEval.evaluate()
        self.cocoEval.accumulate()
        self.results = self.cocoEval.eval
        self.cocoEval.summarize()
        return

    def _get_conf_list(self, result):
        conf_list = []
        for idx in result['gtMatches'][0, :]:
            if idx == 0:
                conf_list.append(0)
            else:
                gt_idx = np.where(result['dtIds'] == idx)
                conf_list.append(float(np.array(result['dtScores'])[gt_idx]))
        return conf_list

    def PRF_df(self):
        '''
        Returns: catId 별, confidence[0:.01:1] 별  DataFrame
        '''
        cat_list = []
        p_list = []
        r_list = []
        conf_list = []
        for catId in range(1, self.cat_nums + 1):
            for conf in np.linspace(0, 1, 101):
                p, r = self.PR(conf, catId=catId)
                p_list.append(p)
                r_list.append(r)
                cat_list.append(catId)
                conf_list.append(conf)

        df = pd.DataFrame(
            {'precision': p_list, 'recall': r_list, 'category': cat_list, 'confidence': conf_list})
        df['category'] = df['category'].astype('str')
        df['category'] = df['category'].apply(
            lambda x: self.catId_to_name[int(x)])
        df['f1_score'] = 2 * (df['precision'] * df['recall']) / \
            (df['precision'] + df['recall'])

        return df


# if __name__ == '__main__':
#     coco_visual = COCO_converter('../info/test.json', 'result.json')
#     print(coco_visual.visual(idx=516))
#     print(coco_visual.coco_gt.getImgIds(catIds=1))

#     coco_visual.visual(idx=1214)
#     coco_visual
