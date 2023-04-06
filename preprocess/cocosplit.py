import json
import funcy
import sklearn.model_selection as ms
import pandas as pd
import numpy as np
import os
import argparse

parser = argparse.ArgumentParser(
    description='coco_file을 가지고 train test를 나누는 script'
)

parser.add_argument('--coco_path')
parser.add_argument('--train_size', default=0.8, type=float)
parser.add_argument('--save_path')

args = parser.parse_args()

with open(args.coco_path, 'r') as json_file:
    json_data = json.load(json_file)
data = pd.DataFrame(json_data['images'])
x_train, x_test = ms.train_test_split(data, train_size=args.train_size)

print('train 개수 :', len(x_train))
print('test 개수 :', len(x_test))

# Filter the corresponding image from all images
train_anns = funcy.lfilter(lambda x: x['image_id'] in np.array(
    x_train['id']), json_data['annotations'])

test_anns = funcy.lfilter(lambda x: x['image_id'] in np.array(
    x_test['id']), json_data['annotations'])


# make train_data_df
train_df = {}
train_df['images'] = x_train.to_dict('records')
train_df['annotations'] = train_anns
train_df['categories'] = json_data['categories']

test_df = {}
test_df['images'] = x_test.to_dict('records')
test_df['annotations'] = test_anns
test_df['categories'] = json_data['categories']


# save train test json file
with open(os.path.join(args.save_path, "train.json"), "w") as json_file:
    json.dump(train_df, json_file, indent=4)

with open(os.path.join(args.save_path, "test.json"), "w") as json_file:
    json.dump(test_df, json_file, indent=4)
