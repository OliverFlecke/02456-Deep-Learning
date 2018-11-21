# https://arxiv.org/pdf/1808.09001.pdf
# https://www.learnopencv.com/installing-deep-learning-frameworks-on-ubuntu-with-cuda-support/
# https://www.learnopencv.com/keras-tutorial-fine-tuning-using-pre-trained-models/

import os
import numpy as np
import pandas as pd
import matplotlib.pyplot as plt

import tensorflow as tf
from keras import models, layers, optimizers, losses, applications
from keras.utils.generic_utils import CustomObjectScope
from keras.preprocessing.image import ImageDataGenerator
from sklearn.metrics import average_precision_score as apscore
from sklearn.metrics import precision_score as pscore

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

class RNet(models.Sequential):
    """
        Using pre-trained ResNet50 without head
    """
    def __init__(self):
        super(RNet, self).__init__()

        sizes = [(200,200,3), (7,7,2048), 1024, 37]

        # Load the pre-trained base model
        base = applications.ResNet50(
            weights='imagenet',
            include_top=False,
            input_shape=sizes[0])
        # Freeze the layers except the last ones
        # for layer in base.layers[:-4]:
        #     layer.trainable = False

        self.add(base)
        self.add(layers.Flatten())
        self.add(layers.Dense(sizes[-2], activation='relu'))
        self.add(layers.Dense(sizes[-1], activation='sigmoid'))
# rnet = RNet()
# rnet.summary()

# https://keras.io/preprocessing/image/#imagedatagenerator-class


generator_args = {
    'rescale':1./255,
    # 'validation_split':0.3,
    # 'rotation_range':45,
    # 'width_shift_range':0.2,
    # 'height_shift_range':0.2,
    # 'horizontal_flip':True,
    # 'vertical_flip':True,
}

generator = ImageDataGenerator(**generator_args)

flow_args = {
    # 'directory': '../dataset_v2/train/classes',
    'target_size':(128, 128), #(200, 200) for ResNet50
    'batch_size':32,
    # 'shuffle':True,
}

# general_class_train_gen = generator.flow_from_directory(subset='training',   directory='../dataset_v2/train/general_class', **flow_args)
# general_class_valid_gen = generator.flow_from_directory(subset='validation', directory='../dataset_v2/train/general_class', **flow_args)

large_vehicle_train_gen = generator.flow_from_directory(subset='training',   directory='../dataset_v2/train/large_vehicle', **flow_args)
large_vehicle_valid_gen = generator.flow_from_directory(subset='validation', directory='../dataset_v2/train/large_vehicle', **flow_args)

# small_vehicle_gen = generator.flow_from_directory(directory='../dataset_v2/train/small_vehicle', **flow_args)
# color_gen         = generator.flow_from_directory(directory='../dataset_v2/train/color', **flow_args)
# feature_gen       = generator.flow_from_directory(directory='../dataset_v2/train/feature', **flow_args)


train_gen = generator.flow_from_directory(directory='../dataset_v2/train/classes', **flow_args)
# valid_gen = generator.flow_from_directory(directory='../dataset_v2/train/cropped', **flow_args, shuffle=False)
test_gen = generator.flow_from_directory(directory='../dataset_v2/test/cropped/test3', **flow_args, shuffle=False)

train_gen = large_vehicle_train_gen
valid_gen = large_vehicle_valid_gen

from collections import Counter
Nm = np.array(sorted(Counter(train_gen.classes).items()))[:,1]
w0 = np.maximum(Nm.astype(np.float)/train_gen.samples, 0.1)
w = (1 - w0) / w0
const = -1/(Nm.sum() + len(w))

def WeightedCELoss(ys, ts):
    return const * (w * ys * tf.log(ts) + (1 - ys) * tf.log(1 - ts))

compile_args = {
    'optimizer':optimizers.SGD(lr=0.002, momentum=0.9), # 0.01 / 0.002
    'loss': WeightedCELoss, #losses.binary_crossentropy
    'metrics':['acc']
}
fit_gen_args = {
    'generator':train_gen,
    # 'validation_data':valid_gen,
    'steps_per_epoch':len(train_gen),
    # 'validation_steps':len(valid_gen),
    'epochs':100,
    'class_weight':None
}

# model = MNet(len(train_gen.class_indices), 'softmax')
model = MNet()
model.summary()
model.compile(**compile_args)
model.fit_generator(**fit_gen_args)
model.save('mnet150WEC.h5')

# get_tag_ids = lambda x: int(re.split(r'[\_.]', x)[-2])
# tag_ids = list(map(get_tag_ids, valid_gen.filenames))
# train = train.reset_index('image_id',drop=True)
# targets = train.ix[tag_ids]
model = MNet()
model.load_weights('../mnet100.h5')

# model = models.load_model('../mnet100.h5', custom_objects={'MNet':MNet})

# Get predictions
predictions_proba = model.predict_generator(valid_gen, steps=len(valid_gen))
# predictions = predictions_proba > 0.1

predictions = list(map(np.argmax, predictions_proba))
targets = valid_gen.classes

AP = np.zeros(len(valid_gen.class_indices))
for category in np.unique(valid_gen.classes):
    ts = valid_gen.classes==category
    ys = np.array(predictions) == category
    AP[category] = APScore(ys, ts)
AP
np.mean(AP)
    
np.save('predictions_proba.npy', predictions_proba)
np.save('predictions.npy', predictions)
predictions = np.load('predictions.npy')

test_gen.filenames[:5]
test_predictions_proba = model.predict_generator(test_gen, steps=len(test_gen))
# np.save('test_predictions_proba.npy', test_predictions_proba)

# Get targets
train = pd.read_csv('../dataset_v2/train.csv')
train = train.replace(' ', '_', regex=True)
train = train.replace('/', '_', regex=True)
train = train.drop(['p1_x', 'p_1y', ' p2_x', ' p2_y', ' p3_x', ' p3_y', ' p4_x', ' p4_y'], axis=1)
for category in ['general_class', 'sub_class', 'color']:
    train[category] = train[category].astype('category')
train = pd.get_dummies(train, prefix='', prefix_sep='')
train = train.groupby(['image_id', 'tag_id']).first()
train = train.reindex(sorted(train.columns), axis=1)
targets = (train.values == 1) # Remove -1s

def APScore(ys, ts):
    TP = 0; FP = 0; total = 0
    for (y,t) in zip(ys,ts):
        TP += y and t
        FP += y and not t
        total += (TP / (TP + FP)) if y and t else 0
    # for k in range(1,1+len(ts)):
    #     total += pscore(ts[:k], ys[:k]) * ts[k-1]
    return total / sum(ts)

assert(predictions.shape == targets.shape)
N, M = targets.shape
AP = np.zeros(M)
for category in range(M):
    AP[category] = APScore(predictions[:,category], targets[:,category])
MAP = np.mean(AP)

MAP
AP
