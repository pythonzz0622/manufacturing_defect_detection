import json
import funcy
import sklearn.model_selection as ms
import pandas as pd
import numpy as np
import os

with open('../info/coco.json', 'r') as json_file:
    json_data = json.load(json_file)
data = pd.DataFrame(json_data['images'])
x_train, x_test = ms.train_test_split(data, train_size=0.8)

print('train 개수 :', len(x_train))
print('test 개수 :', len(x_test))

train_anns = funcy.lfilter(lambda a: a['image_id'] in np.array(
    x_train['id']), json_data['annotations'])

test_anns = funcy.lfilter(lambda a: a['image_id'] in np.array(
    x_test['id']), json_data['annotations'])
json_data['categories']
train_df = {}
train_df['images'] = x_train.to_dict('records')
train_df['annotations'] = train_anns
train_df['categories'] = json_data['categories']

test_df = {}
test_df['images'] = x_test.to_dict('records')
test_df['annotations'] = test_anns
test_df['categories'] = json_data['categories']


with open("../info/train.json", "w") as json_file:
    json.dump(train_df, json_file, indent=4)

with open("../info/test.json", "w") as json_file:
    json.dump(test_df, json_file, indent=4)
