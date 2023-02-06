# python tools/test.py custom_configs/retinanet_swin-t_ms/final.py \
# work_dirs/final/latest.pth \
# --show-dir work_dirs/result


python tools/test.py custom_configs/retinanet_swin-l_ms/retinanet_swin-l-run.py \
work_dirs/retinanet_swin-l-run/latest.pth \
--show-dir work_dirs/result


# python tools/test.py custom_configs/retinanet_swin-l_ms/retinanet_swin-l-run.py \
# work_dirs/final/latest.pth \
# --format-only --eval-options "jsonfile_prefix=./result"