# https://arxiv.org/pdf/1808.09001.pdf
# https://www.learnopencv.com/installing-deep-learning-frameworks-on-ubuntu-with-cuda-support/
# https://www.learnopencv.com/keras-tutorial-fine-tuning-using-pre-trained-models/
# https://keras.io/preprocessing/image/#imagedatagenerator-class

import os
import math
import numpy as np
import pandas as pd
import matplotlib.pyplot as plt

from collections import Counter
from keras import backend as K
from keras import models, layers, optimizers, losses, applications, callbacks, metrics
from keras.preprocessing.image import ImageDataGenerator

class MNet(models.Sequential):
    """
        Using pre-trained MobileNet without head
    """
    def __init__(self, outputs=37, output_layer='sigmoid', **kwargs):
        super(MNet, self).__init__()

        sizes = [(128,128,3), (4,4,1024), 1024, outputs]

        # Load the pre-trained base model
        base = applications.MobileNet(
            weights='imagenet',
            include_top=False,
            input_shape=sizes[0])
        # Freeze the layers except the last ones
        # for layer in base.layers[:-4]:
        #     layer.trainable = False

        self.add(base)
        self.add(layers.Flatten())
        self.add(layers.Dense(sizes[-2], activation='relu'))
        self.add(layers.Dense(sizes[-1], activation=output_layer))


def train_classifier(categories, source_dir, target_dir):
    # Convert singleton to list
    if not isinstance(categories, (list,)):
        categories = [categories]

    generator_args = {
        'validation_split':0.2,
        'rotation_range':45,
        'width_shift_range':0.2,
        'height_shift_range':0.2,
        'horizontal_flip':True,
        'vertical_flip':True,
        'fill_mode':'constant',
        'cval':255
    }
    generator = ImageDataGenerator(**generator_args)

    # Save history
    history = {}

    for category in categories:
        print(f'Training {category}')
        
        flow_args = {
            'target_size':(128, 128),
            'batch_size':32,
            'directory': f'{source_dir}{category}',
        }
        
        train_gen = generator.flow_from_directory(subset='training', **flow_args)
        valid_gen = generator.flow_from_directory(subset='validation', **flow_args)
        
        w = get_class_weight(train_gen) 
        outputs = len(train_gen.class_indices)
        samples = train_gen.samples + valid_gen.samples
        epochs = int(1e4 / np.sqrt(samples))
        lr = np.sqrt(samples) / 1e4
        patience = epochs // 5

        model_args = {
            'outputs':outputs,
            'output_layer':'softmax'
        }
    
        compile_args = {
            'optimizer':optimizers.SGD(
                lr=0.002, 
                momentum=0.9, 
                decay=0.0),
            'loss':losses.binary_crossentropy,
            'metrics':['acc']
        }

        fit_gen_args = {
            'generator':train_gen,
            'validation_data':valid_gen,
            'steps_per_epoch':len(train_gen),
            'validation_steps':len(valid_gen),
            'epochs':epochs,
            'class_weight':dict(enumerate(w)),
            'callbacks':[
                callbacks.EarlyStopping(
                    monitor='val_acc', 
                    min_delta=1e-4, 
                    patience=patience, 
                    restore_best_weights=True),
                # callbacks.ModelCheckpoint(
                #     filepath=f'{target_dir}wce-{category}-checkpoint.hdf5', 
                #     monitor='val_acc', 
                #     save_weights_only=True,
                #     save_best_only=True)
            ],
        }

        model = MNet(**model_args)    
        model.compile(**compile_args)
        history[category] = model.fit_generator(**fit_gen_args)
        model.save_weights(f'{target_dir}wce-{category}.h5')

    return history

def pc_sleep():
    os.system("rundll32.exe powrprof.dll,SetSuspendState 0,1,0")

def adjust_ylim(ymin, ymax, precision=1e3):
    return math.floor(ymin*precision)/precision, \
           math.ceil (ymax*precision)/precision

def plot_history(history):

    num_plots = len(history)
    rows = math.floor(math.sqrt(num_plots))
    cols = math.ceil(num_plots/rows)

    plt.figure(figsize=(30,20))
    for i, (c, h) in enumerate(history.items()):
        plt.subplot(rows,cols,i+1)
        plt.plot(h.history['val_acc'])
        plt.title(c)
        plt.ylim(adjust_ylim(*plt.ylim()))
    plt.show()

def get_class_weight(generator):
    Nm = np.array(sorted(Counter(generator.classes).items()))[:,1]
    w0 = np.maximum(Nm.astype(np.float)/generator.samples, 0.1)
    return (1 - w0) / w0


def K_MAP(y_true, y_pred):
    return K.mean(K.map_fn(K_AP, *(y_true, y_pred)))

def K_AP(y_true, y_pred):
    ts = K.in_top_k(y_pred, y_true, K.sum(y_true))

    return 

def MAP(y_true, y_pred, columns):
    APs = list(map(AP, y_true.T, y_pred.T))
    return dict(zip(columns,APs)), np.mean(APs)

def AP(y_true, y_pred):
    # Sort y_true by y_pred and get K first elems
    ts = [t for y,t in sorted(zip(y_pred,y_true))]
    ts = np.array(ts[:np.sum(y_true)])
    TP = np.cumsum(ts)
    FP = np.cumsum(1-ts)
    return np.mean((TP / (TP + FP)) * ts)

    
# K.mean(K.metrics.sparse_average_precision_at_k(K.cast(y_true, K.int64), y_pred, 1)[0])

# numpy.mean(numpy.cumsum(y_true[numpy.argsort(~y_pred)]) / numpy.cumsum(y_true[numpy.argsort(~y_true)]))

source_dir = '../dataset_v2/train/classes/'
category = ''
target_dir = '../models/divided/'
# history = train_classifier(os.listdir(source_dir), source_dir, target_dir)
# plot_history(history)

generator_args = {
    'validation_split':0.2,
    'rotation_range':45,
    'width_shift_range':0.2,
    'height_shift_range':0.2,
    'horizontal_flip':True,
    'vertical_flip':True,
    'fill_mode':'constant',
    'cval':255
    
}
generator = ImageDataGenerator(**generator_args)
        
flow_args = {
    'target_size':(128, 128),
    'batch_size':32,
    'directory': f'{source_dir}{category}',
    # 'save_to_dir':'../dataset_v2/train/saved'
}

train_gen = generator.flow_from_directory(subset='training', **flow_args)
valid_gen = generator.flow_from_directory(subset='validation', **flow_args)

test_gen = ImageDataGenerator().flow_from_directory(
    directory='../dataset_v2/test/', 
    shuffle=False, 
    target_size=(128, 128), 
    batch_size=32
)

w = get_class_weight(train_gen)
_EPSILON = K.epsilon()

def weighted_crossentropy(y_true, y_pred):
    y_pred = K.clip(y_pred, _EPSILON, 1.0-_EPSILON)
    out = -(K.constant(w) * y_true * K.log(y_pred) + (1.0 - y_true) * K.log(1.0 - y_pred))
    return K.mean(out, axis=-1)

samples = train_gen.samples + valid_gen.samples
epochs = int(1e4 / np.sqrt(samples))
lr = np.sqrt(samples) / 1e4
patience = epochs // 10

model_args = {
    # 'outputs':len(train_gen.class_indices),
    # 'output_layer':'softmax'
}

compile_args = {
    'optimizer':optimizers.SGD(
        lr=0.002, 
        momentum=0.9, 
        decay=0.00),
    # 'loss':losses.binary_crossentropy,
    'loss':weighted_crossentropy,
    'metrics':['acc']
}

fit_gen_args = {
    'generator':train_gen,
    'validation_data':valid_gen,
    'steps_per_epoch':len(train_gen),
    'validation_steps':len(valid_gen),
    'epochs':50,
    # 'class_weight':dict(enumerate(w)),
    'callbacks':[
        callbacks.EarlyStopping(
            monitor='val_loss', 
            min_delta=1e-4, 
            patience=10, 
            restore_best_weights=True),
        # callbacks.ModelCheckpoint(
        #     filepath='../models/divided/general_class-checkpoint.hdf5', 
        #     monitor='val_acc', 
        #     save_weights_only=True,
        #     save_best_only=True)
    ],
}

model_path = '../models/mnet50aug-wce-2.h5'

model = MNet(**model_args)    
model.compile(**compile_args)
# model.load_weights(model_path)
model.fit_generator(**fit_gen_args)
model.save_weights(model_path)


## PREDICTIONS

proba = model.predict_generator(test_gen, steps=len(test_gen))
cols = os.listdir(f'{source_dir}{category}')
cols = [x.replace('_', ' ') for x in cols]
cols = sorted(pd.read_csv('../dataset_v2/answer_template.csv').columns)

# VALIDATION

test_gen = ImageDataGenerator().flow_from_directory(
    directory='../dataset_v2/train/', 
    classes=['cropped'],
    shuffle=False, 
    target_size=(128, 128), 
    batch_size=32,
)

# y_true, cols = get_targets(test_gen)
y_pred = model.predict_generator(test_gen, steps=len(test_gen))
MAP(y_true, y_pred, cols)
valid_gen.filenames[:5]


def eval_pred(pred):
    answers = get_answers(pred, dataset='train')
    df = pd.read_csv('../dataset_v2/train.csv')
    for category in ['general_class', 'sub_class', 'color']:
        df[category] = df[category].astype('category')
    df = pd.get_dummies(df, prefix='', prefix_sep='')
    df = df.groupby('tag_id').first()
    df = df.drop(df.columns[:9], axis=1) 
    df = df.reindex(sorted(df.columns), axis=1)
    df = df.replace(-1, 0)

    for col in answers.columns:
        for idx in answers.index:
            answers.at[idx, col] = df.at[answers.at[idx, col], col]
    
    AP = {}
    for col in answers.columns:
        num_true = sum(answers[col])
        ts = np.array(answers[col][:num_true])
        TP = np.cumsum(ts)
        FP = np.cumsum(1-ts)
        AP[col] = np.mean((TP / (TP + FP)) * ts)
    
    return AP, np.mean(list(AP.values()))


    
def get_targets(gen):
    tag_ids = [int(x.split('_')[-1][:5]) for x in test_gen.filenames]
    df = pd.read_csv('../dataset_v2/train.csv')
    for category in ['general_class', 'sub_class', 'color']:
        df[category] = df[category].astype('category')
    df = pd.get_dummies(df, prefix='', prefix_sep='')
    df = df[df.tag_id.isin(tag_ids)]
    df = df.groupby('tag_id').first()
    df = df.drop(df.columns[:9], axis=1)    
    df = df.reindex(sorted(df.columns), axis=1)
    df = df.replace(-1, 0)
    return np.array([df.loc[t].values for t in tag_ids]), df.columns


import pandas as pd
answer = pd.read_csv('../dataset_v2/aug.csv')
colors = pd.read_csv('../dataset_v2/colors.csv')

columns = [x.replace('_', '/') for x in os.listdir('../dataset_v2/train/divided/color')]
for color in columns:
    answer[color] = colors[color]



import json
colors = json.load(open('../dataset_v2/colors.json', 'r'))
offline = json.load(open('../dataset_v2/offline.json', 'r'))
base   = json.load(open('../dataset_v2/base.json', 'r'))
aug    = json.load(open('../dataset_v2/aug.json', 'r'))
colors_csv = pd.read_csv('../dataset_v2/colors.csv')
base_csv   = pd.read_csv('../dataset_v2/base.csv')
aug_csv    = pd.read_csv('../dataset_v2/aug.csv')
csvs = [aug_csv, base_csv, colors_csv]

answers = pd.read_csv('../dataset_v2/answer_template.csv')

for key in colors.keys():
    vals = aug[key], base[key], colors[key]
    idx = np.argmax(vals)
    answers[key] = csvs[idx][key]

answers.to_csv('../dataset_v2/answer.csv', index=False)


base[columns]

def print_color_table():
    float_formatter = lambda x: "%.2f" % x
    print('\\begin{tabular}{l|c|c|c|c}')
    print('Color \t\t& Base \t\t\t& Augmented \t& Color Offline \t& Color Online \t\\\\ \\hline')
    for col in columns:
        scores = np.array([base[col], aug[col], offline[col], colors[col]])
        scores = [float_formatter(x) for x in scores]
        print(f'{col.capitalize()} \t\t& {scores[0]} \t\t& {scores[1]} \t\t& {scores[2]} \t\t\t& {scores[3]} \t\t\\\\')
    
    
    score_dicts = [base, aug, offline, colors]
    score_dicts = [[score_dict[col] for col in columns] for score_dict in score_dicts]
    means = [np.mean(score_dict) for score_dict in score_dicts]
    means = [float_formatter(x) for x in means]
    print(f'Mean score \t\t& {means[0]} \t\t& {means[1]} \t\t& {means[2]} \t\t& {means[3]}')
    print('\\end{tabular}')