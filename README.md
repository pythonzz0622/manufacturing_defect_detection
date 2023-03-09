# manufacturing_defect_detection

Python: 3.8.13
CPU : AMD Ryzen Threadripper 3990X 64-Core Processor
GPU 0: GeForce RTX 3090
Torch:1.12.1
TorchVision: 0.13.1
OpenCV: 4.6.0
MMCV: 1.7.0
MMDetection: 2.26.0



data preprocessing.ipynb

-> folder 옮긴후에  coco data format으로 맞춰서 만들기

data_visualization.ipynb

-> json file 형식 시각화 & coco format 형식 시각화


mv defect to normal.ipynb

-> labelImg를 활용해 1차 이미지 정제 후에 data normal, defect 다시 분기


remove anomaly data.ipynb

-> t-test로 hist 분포 파악 후 이미지 유사도로 이상치 데이터 제거


mmdetection model list 

