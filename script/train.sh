#!/bin/bash
cd /home/user304/users/jiwon/graduation_paper/mmdetection/
# python tools/train.py custom_configs/retinanet_swin-t_ms/final.py
nohup python tools/train.py custom_configs/retinanet_swin-l_ms/run.py --work-dir ckpts/wan-retinanet_swin-l >> ./log/$(date +%m%d_%H%M).log &
# python tools/train.py custom_configs/yolox/run.py --work-dir ckpts/yolox
# python tools/train.py custom_configs/faster_rcnn/rcnn.py --work-dir ckpts/faster_rcnn