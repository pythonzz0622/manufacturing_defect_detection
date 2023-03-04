# python tools/test.py custom_configs/retinanet_swin-t_ms/final.py \
# work_dirs/final/latest.pth \
# --show-dir work_dirs/result


python tools/test.py custom_configs/retinanet_swin-l_ms/run.py \
ckpts/new_retinanet_swin-l/latest.pth \
--show-dir result/new_retinanet_swin-l \
--format-only --eval-options "jsonfile_prefix=./result/new_retinanet_swin-l/result"
# --eval bbox


# python tools/test.py custom_configs/retinanet_swin-l_ms/run.py \
# ckpts/retinanet_swin-l/epoch_50.pth \
# --format-only --eval-options "jsonfile_prefix=./result/retinanet_swin-l/result" 

# python tools/test.py custom_configs/faster_rcnn/rcnn.py \
# ckpts/faster_rcnn/epoch_10.pth \
# --format-only --eval-options "jsonfile_prefix=./result/faster_rcnn/result" 