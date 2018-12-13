import os
import numpy as np
import pandas as pd

def get_pred_index(dataset='test'):
    df = pd.read_csv(f'../dataset_v2/{dataset}.csv')
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

def get_answers(proba, dataset='test'):
    pred = pd.DataFrame(proba, index=get_pred_index(dataset=dataset), columns=get_pred_columns())

    for col in pred.columns:
        pred[col] = pred[col].sort_values(ascending=False).index

    return pred.reset_index(drop=True)

def create_partial_answers(proba, cols):
    pred = pd.DataFrame(proba, index=get_pred_index(), columns=cols)

    answers = pd.read_csv('../dataset_v2/answer_template.csv')
    for col in pred.columns:
        answers[col] = pred[col].sort_values(ascending=False).index

    for col in set(answers.columns) - set(pred.columns):
        answers[col] = answers[pred.columns[0]]

    answers.to_csv('../dataset_v2/answer.csv', index=False)

def create_answers(test_predictions_proba):
    predictions = pd.DataFrame(test_predictions_proba,
        index=get_pred_index(), columns=get_pred_columns())

    answers = pd.read_csv('../dataset_v2/answer_template.csv')
    for column in answers.columns:
        answers[column] = predictions[column].sort_values(ascending=False).index

    answers.to_csv('../dataset_v2/answer.csv', index=False)
