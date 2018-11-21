import os
import numpy as np
import pandas as pd

def get_pred_index():
    df = pd.read_csv('../dataset_v2/test.csv')
    df = df.groupby(['image_id', 'tag_id']).first()
    df = df.reset_index('image_id', drop=True)
    return df.index

def get_pred_columns():
    df = pd.read_csv('../dataset_v2/train.csv')
    df = df.drop(df.columns[:10], axis=1)
    for category in df.columns[df.dtypes == object]:
        df[category] = df[category].astype('category')
    df = pd.get_dummies(df, prefix='', prefix_sep='')
    return sorted(df.columns)

def create_answers(test_predictions_proba):
    predictions = pd.DataFrame(test_predictions_proba,
        index=get_pred_index(), columns=get_pred_columns())

    answers = pd.read_csv('../dataset_v2/answer_template.csv')
    for column in answers.columns:
        answers[column] = predictions[column].sort_values(ascending=False).index

    answers.to_csv('../dataset_v2/answer.csv', index=False)
