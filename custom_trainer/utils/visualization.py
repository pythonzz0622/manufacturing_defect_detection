import utils.type_converter as t_cvt
import cv2
import matplotlib.pyplot as plt
import numpy as np
import pandas as pd


class VisualTool:
    ''' Object detection 시각화 툴
    Args:
        coco_gt_path (str) : ground_truth coco path 
        coco_pred_path (str) : model prediciton coco path
        model_name (str) : using model name
    '''

    def __init__(self, coco_gt_path, coco_pred_path=None, model_name='swin'):
        self.coco_cvt = t_cvt.COCO_converter(coco_gt_path, coco_pred_path)
        self.model_name = model_name

    def visual(self, idx, img_prefix='../dataset/'):
        ''' coco idx별 시각화

        Args:
            idx (int) : coco img idx
        '''
        img_annos = self.coco_cvt.img_annos(idx)
        img = cv2.imread(img_prefix + img_annos['file_name'])
        img = cv2.cvtColor(img, cv2.COLOR_BGR2RGB)
        font = cv2.FONT_HERSHEY_SIMPLEX
        palette = (0, 255, 0)
        if img_annos.get('ground_truth'):
            gts = img_annos['ground_truth']
            for i, catId in enumerate(gts['category_id']):
                anno = t_cvt.coco_to_cv2(gts['bbox'][i], dim=1)
                cv2.rectangle(img, (anno[0], anno[1]),
                              (anno[2], anno[3]), palette,  thickness=1)
                cv2.putText(
                    img, f'{catId}', (anno[0], anno[1] - 5), font, 0.8, palette, thickness=2)

        if img_annos.get('preds'):
            palette = (0, 255, 255)
            preds = img_annos['preds']
            for i, catId in enumerate(preds['category_id']):
                anno = t_cvt.coco_to_cv2(preds['bbox'][i], dim=1)
                score = round(preds['score'][i], 2)
                cv2.rectangle(img, (anno[0], anno[1]),
                              (anno[2], anno[3]), palette,  thickness=1)
                cv2.putText(
                    img, f'{catId} : {score}', (anno[0], anno[1] - 5), font, 0.8, palette, thickness=2)
        plt.figure(figsize=(15, 15))
        plt.title(f'idx {idx}')
        plt.imshow(img)
        plt.show()
        return

    def PR_plot(self):
        # Precision_Recall Curve Plot
        df = self.coco_cvt.PRF_df()
        for i, category in enumerate(df['category'].unique()):
            precision = self.coco_cvt.results['precision'][0,
                                                           :, i, 0, -1].mean()
            data = df[df['category'] == category]
            plt.plot(data['recall'], data['precision'],
                     label=f'{category} {np.round(precision , 3)}')
            plt.legend()
            plt.xlabel('recall')
            plt.ylabel('precision')

        mAP = self.coco_cvt.results['precision'][0, :, :, 0, -1].mean()
        plt.title(
            f"mAP@0.5 : {np.round(mAP , 2)}")
        # plt.savefig(f'./result/{self.model_name}_PR_curve.png', dpi=300)
        plt.show()
        return df

    def conf_vs(self):
        # confidence 별 recall , precision , f1 score plot 반환
        df = self.coco_cvt.PRF_df()
        for j in ['recall', 'precision', 'f1_score']:
            for i, category in enumerate(df['category'].unique()):
                data = df[df['category'] == category]
                plt.plot(np.linspace(0, 1, 101), data[j], label=f'{category}')
                plt.legend()
                plt.xlabel('Confidence')
                plt.ylabel(j)
            plt.title(f'{self.model_name} conf vs {j}')
            # plt.savefig(
            #     f'./result/{self.model_name}_conf_{j[0]}curve.png', dpi=300)
            plt.show()
