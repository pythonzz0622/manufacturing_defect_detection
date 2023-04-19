MODEL_NAME=retina-pafpn-swin-l
nohup python train.py --save_name ${MODEL_NAME} --epochs 20 \
--interval 100 \
--train_path ./dataset/train.json  \
--val_path ./dataset/test.json \
>> log/train_${MODEL_NAME}_$(date +%m%d_%H%M).log &