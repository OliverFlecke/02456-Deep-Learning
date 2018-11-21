import os
import numpy as np
import pandas as pd

test = pd.read_csv('../dataset_v2/test.csv')
test = test.groupby(['image_id', 'tag_id']).first()
test = test.reset_index('image_id', drop=True)

train = pd.read_csv('../dataset_v2/train.csv')
train = train.drop(train.columns[:10], axis=1)
for category in ['general_class', 'sub_class', 'color']:
    train[category] = train[category].astype('category')
train = pd.get_dummies(train, prefix='', prefix_sep='')

test_predictions_proba = np.load('test_predictions_proba.npy')

predictions = pd.DataFrame(test_predictions_proba,
    index=test.index, columns=sorted(train.columns))

answers = pd.read_csv('../dataset_v2/answer_template.csv')
for column in answers.columns:
    answers[column] = predictions[column].sort_values(ascending=False).index

answers.to_csv('../dataset_v2/answer.csv', index=False)
