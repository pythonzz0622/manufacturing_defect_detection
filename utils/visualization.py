from utils import type_converter as tcvt
import cv2
import matplotlib.pyplot as plt
import numpy as np
import pandas as pd
from ipywidgets import interact
import wandb
import torch
import os.path as osp
class VisualTool:
    ''' Object detection 시각화 툴
    Args:
        coco_gt_path (str) : ground_truth coco path 
        coco_pred_path (str) : model prediciton coco path
        model_name (str) : using model name
    '''

    def __init__(self, coco_gt_path, coco_pred_path=None, model_name='swin'):
        self.coco_cvt = tcvt.COCO_converter(coco_gt_path, coco_pred_path)
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
                anno = tcvt.coco_to_cv2(gts['bbox'][i], dim=1)
                cv2.rectangle(img, (anno[0], anno[1]),
                              (anno[2], anno[3]), palette,  thickness=1)
                cv2.putText(
                    img, f'{catId}', (anno[2], anno[3] + 5), font, 0.8, palette, thickness=2)

        if img_annos.get('preds'):
            palette = (0, 255, 255)
            preds = img_annos['preds']
            for i, catId in enumerate(preds['category_id']):
                anno = tcvt.coco_to_cv2(preds['bbox'][i], dim=1)
                score = round(preds['score'][i], 2)
                cv2.rectangle(img, (anno[0], anno[1]),
                              (anno[2], anno[3]), palette,  thickness=1)
                cv2.putText(
                    img, f'{catId} : {score}', (anno[2]-100, anno[3] - 5), font, 0.8, palette, thickness=2)
        plt.figure(figsize=(8, 8))
        plt.title(f'idx {idx}')
        plt.imshow(img)
        plt.axis('off')
        plt.show()
        return


    def PR_plot(self):
        # Precision_Recall Curve Plot
        df = self.coco_cvt.PRF_df()
        fig , ax = plt.subplots(1,1)
        for i, category in enumerate(df['category'].unique()):
            precision = self.coco_cvt.results['precision'][0,
                                                           :, i, 0, -1].mean()
            data = df[df['category'] == category]
            ax.plot(data['recall'], data['precision'],
                     label=f'{category} {np.round(precision , 3)}')
            ax.legend()
            ax.set_xlabel('recall')
            ax.set_ylabel('precision')

        mAP = self.coco_cvt.results['precision'][0, :, :, 0, -1].mean()
        ax.set_title(
            f"mAP@0.5 : {np.round(mAP , 2)}")
        # plt.savefig(f'./result/{self.model_name}_PR_curve.png', dpi=300)
        # plt.show()
        return fig

    def conf_vs(self):
        # confidence 별 recall , precision , f1 score plot 반환
        df = self.coco_cvt.PRF_df()
        figs = []
        for j in ['recall', 'precision', 'f1_score']:
            fig , ax = plt.subplots(1,1,)
            for i, category in enumerate(df['category'].unique()):
                data = df[df['category'] == category]
                ax.plot(np.linspace(0, 1, 101), data[j], label=f'{category}')
                ax.legend()
                ax.set_xlabel('Confidence')
                ax.set_ylabel(j)
            ax.set_title(f'{self.model_name} conf vs {j}')
            # plt.savefig(
            #     f'./result/{self.model_name}_conf_{j[0]}curve.png', dpi=300)
            figs.append(fig)
        return figs



def bbox_debugger(img_annos , catId_to_name , img_prefix ='' ):
    img = cv2.imread(img_prefix + img_annos['file_name'])
    img = cv2.cvtColor(img , cv2.COLOR_BGR2RGB)
    box_data = []
    if img_annos.get('preds'):
        for i in range(len(img_annos['preds']['category_id'])):
            class_id = img_annos['preds']['category_id'][i]
            bbox = tcvt.coco_to_cv2(img_annos['preds']['bbox'][i] , dim= 1)
            score = img_annos['preds']['score'][i]
            bbox = {
            # another box expressed in the pixel domain
            "position": {"minX": bbox[0], "maxX": bbox[2], "minY": bbox[1], "maxY": bbox[3]},
            "domain": "pixel",
            "class_id": class_id,
            "box_caption": f"{catId_to_name[class_id]} , {score}",
            "scores": {'score' : score}
            }
            box_data.append(bbox)
    if img_annos.get('ground_truth'):
        box_data_gt = []
        for i in range(len(img_annos['ground_truth']['category_id'])):
            class_id = img_annos['ground_truth']['category_id'][i]
            bbox = tcvt.coco_to_cv2(img_annos['ground_truth']['bbox'][i] , dim= 1)
            bbox = {
            # another box expressed in the pixel domain
            "position": {"minX": bbox[0], "maxX": bbox[2], "minY": bbox[1], "maxY": bbox[3]},
            "domain": "pixel",
            "class_id": class_id,
            "box_caption": f"{catId_to_name[class_id]} "
            }
            box_data_gt.append(bbox)
        boxes = {"predictions": {"box_data": box_data, "class_labels": catId_to_name},
            "ground_truths": {"box_data": box_data_gt, "class_labels": catId_to_name},}  # inference-space
    else:
        boxes = {"predictions": {"box_data": box_data, "class_labels": catId_to_name}}
    return wandb.Image(img , boxes = boxes)

def input_visual(images , bboxes  ,  batch_size , de_std , de_mean):
    '''
    images (Torch.tensor) : batch별 images
    bboxes (List[Torch.tensor]) : batch 별 bboxes
    de_std : 표준화 전환 
    de_mean : 표준화 전환 
    '''
    batch_size = 8
    ncol = int(batch_size / 2 )
    nrow = 2
    c = 0
    fig , ax = plt.subplots(nrows = nrow , ncols = ncol  , figsize = (12,12))
    for i in range(nrow):
        for j in range(ncol):
            image = images[c].permute(1,2,0)
            img = ((image *  torch.tensor(de_std)) + torch.tensor(de_mean)).int()
            img1 = img.numpy().astype(np.uint8).copy()
            if bboxes[c].size()[0]:
                bbox = bboxes[c].int().numpy()
                for k in range(len(bbox)):
                    img1 = cv2.rectangle(img1 , pt1 =bbox[k,:2]  , pt2 = bbox[k,2:], color = (255,255,0), thickness = 3)
            ax[i,j].set_xticks([])
            ax[i,j].set_yticks([])
            ax[i,j].imshow(img1)
            c += 1
    fig.tight_layout(h_pad = -7)
    return fig


def visual_plot(img_metas , val_nums , epoch , bbox_losses , cls_losses):
    visualtool = VisualTool(coco_gt_path = './dataset/test.json' , coco_pred_path='./result.json' )
    PR_plot = visualtool.PR_plot()
    R_plot , P_plot , F1_plot = visualtool.conf_vs()
    coco_cvt = tcvt.COCO_converter('./dataset/test.json' , 'result.json')
    mAP = coco_cvt.results['precision'][0, :, :, 0, -1].mean()
    cat1_precision ,cat1_recall  = coco_cvt.PR(conf_score=0.5, catId = 1)
    cat2_precision ,cat2_recall = coco_cvt.PR(conf_score=0.5, catId = 2)
    val_log_dict = {
    "val/epoch" : epoch,
    "val/loss_bbox" : bbox_losses / val_nums,
    "val/loss_cls" : cls_losses / val_nums,
    "val/loss_total" : (bbox_losses + cls_losses) / val_nums,
    "val_precision" : (cat1_precision + cat2_precision) /2 ,
    "val_recall" : (cat1_recall + cat2_recall) /2 ,
    "val_mAP" :  mAP}
    wandb.log(val_log_dict)
    debug_img_id_list = [coco_cvt.img_name_to_id[img_meta['filename'].replace('./dataset/images/' ,'')] for img_meta in img_metas]
    val_img_dict =  {
        # 'val/bbox_debug' : wandb_imgs,
        "val/PR_plot": PR_plot,
        "val/R_plot": R_plot,
        "val/P_plot": P_plot,
        "val/F1_plot": F1_plot , 
    }
    try:
        val_img_dict['val/bbox_debug'] = [bbox_debugger(coco_cvt.img_annos(img_id),coco_cvt.catId_to_name ,img_prefix ='./dataset/'  ) for img_id in debug_img_id_list]
    except:
        pass
    wandb.log(val_img_dict)
    plt.close()