python preprocess/label_to_coco.py --label_path ./dataset/labels/ --save_path ./dataset/coco.json
python preprocess/cocosplit.py --coco_path ./dataset/coco.json --save_path ./dataset/