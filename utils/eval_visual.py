import numpy as np
import json
import matplotlib.pyplot as plt
import pandas as pd
from pycocotools.coco import COCO
import re
import os
from pycocotools.coco import COCO
from pycocotools.cocoeval import COCOeval


class eval_plot():
    '''
    confidence score에 따른 plot그리기
    '''

    def __init__(self, coco_gt_path, coco_pred_path, model_name) -> None:
        self.model_name = model_name
        self.coco_gt = COCO(coco_gt_path)
        self.id_to_cat = {str(x['id']): x['name']
                          for x in self.coco_gt.loadCats(self.coco_gt.getCatIds())}
        img_path_to_id = {re.sub('.jpg', '', os.path.basename(
            x['file_name'])): x['id'] for x in self.coco_gt.loadImgs(self.coco_gt.getImgIds())}
        with open(coco_pred_path, 'r') as f:
            result_data = json.load(f)

        if 'yolo' in coco_pred_path:
            for i in range(len(result_data)):
                result_data[i]['image_id'] = img_path_to_id[result_data[i]['image_id']]
                result_data[i]['category_id'] += 1

        coco_pred = self.coco_gt.loadRes(result_data)
        self.cocoEval = COCOeval(self.coco_gt, coco_pred, iouType='bbox')
        self.cocoEval.evaluate()
        self.cocoEval.accumulate()
        self.results = self.cocoEval.eval

    def _get_conf_list(self, result):
        conf_list = []
        for idx in result['gtMatches'][0, :]:
            if idx == 0:
                conf_list.append(0)
            else:
                gt_idx = np.where(result['dtIds'] == idx)
                conf_list.append(float(np.array(result['dtScores'])[gt_idx]))
        return conf_list

    def get_PR(self, conf_score, catId):
        pred_list = []
        truth_list = []
        pred_conf_list = []
        truth_conf_list = []
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
        recall = np.count_nonzero(truth_data) / len(truth_data)
        return precision, recall

    def get_PRF_df(self):
        cat_list = []
        p_list = []
        r_list = []
        conf_list = []
        for catId in range(1, 4):
            for conf in np.linspace(0, 1, 101):
                p, r = self.get_PR(conf, catId=catId)
                p_list.append(p)
                r_list.append(r)
                cat_list.append(catId)
                conf_list.append(conf)

        df = pd.DataFrame(
            {'precision': p_list, 'recall': r_list, 'category': cat_list, 'confidence': conf_list})
        df['category'] = df['category'].astype('str')
        df['category'] = df['category'].apply(lambda x: self.id_to_cat[x])
        df['f1_score'] = 2 * (df['precision'] * df['recall']) / \
            (df['precision'] + df['recall'])

        return df

    def PR_plot(self):
        df = self.get_PRF_df()
        for i, category in enumerate(df['category'].unique()):
            precision = self.results['precision'][0, :, i, 0, -1].mean()
            data = df[df['category'] == category]
            plt.plot(data['recall'], data['precision'],
                     label=f'{category} {np.round(precision , 3)}')
            plt.legend()
            plt.xlabel('recall')
            plt.ylabel('precision')

        mAP = self.results['precision'][0, :, :, 0, -1].mean()
        plt.title(
            f"mAP@0.5 : {np.round(mAP , 2)}")
        plt.savefig(f'./result/{self.model_name}_PR_curve.png', dpi=300)
        plt.show()
        return df

    def conf_vs(self):
        df = self.get_PRF_df()
        for j in ['recall', 'precision', 'f1_score']:
            for i, category in enumerate(df['category'].unique()):
                data = df[df['category'] == category]
                plt.plot(np.linspace(0, 1, 101), data[j], label=f'{category}')
                plt.legend()
                plt.xlabel('Confidence')
                plt.ylabel(j)
            plt.title(f'{self.model_name} conf vs {j}')
            plt.savefig(
                f'./result/{self.model_name}_conf_{j[0]}curve.png', dpi=300)
            plt.show()
