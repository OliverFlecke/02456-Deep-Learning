import os
import numpy as np
import pandas as pd

train = pd.read_csv('../dataset_v2/train.csv')
train = train.replace(' ', '_', regex=True)
train = train.replace('/', '_', regex=True)
train = train.drop(['image_id', 'p1_x', 'p_1y', ' p2_x', ' p2_y', ' p3_x', ' p3_y', ' p4_x', ' p4_y'], axis=1)
for category in ['general_class', 'sub_class', 'color']:
    train[category] = train[category].astype('category')
train = pd.get_dummies(train, prefix='', prefix_sep='')
train = train.groupby('tag_id').first()
train = train.reindex_axis(sorted(train.columns), axis=1)

predictions = np.load('predictions_proba.npy')
predictions = pd.DataFrame(predictions,
    index=train.index.values,
    columns=train.columns.values)

answers = pd.read_csv('../dataset_v2/answer_template.csv')
for column in answers.columns.values:
    answers[column] = predictions[column.replace(' ', '_').replace('/', '_')].sort_values(ascending=False).index.values

answers.to_csv('../dataset_v2/answer.csv', index=False)
