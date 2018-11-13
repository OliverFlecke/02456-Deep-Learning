# https://arxiv.org/pdf/1808.09001.pdf
# https://www.learnopencv.com/installing-deep-learning-frameworks-on-ubuntu-with-cuda-support/
# https://www.learnopencv.com/keras-tutorial-fine-tuning-using-pre-trained-models/

import os
import numpy as np
import pandas as pd
import matplotlib.pyplot as plt

import tensorflow as tf
from keras import models, layers, optimizers, losses, applications
from keras.preprocessing.image import ImageDataGenerator

class MNet(models.Sequential):
    """
        Using pre-trained MobileNet without head
    """
    def __init__(self):
        super(MNet, self).__init__()
        
        sizes = [(128,128,3), (4,4,1024), 1024, 37]

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
        self.add(layers.Dense(sizes[-1], activation='sigmoid'))

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
    # 'rotation_range'=45,
    # 'width_shift_range'=0.2,
    # 'height_shift_range'=0.2,
    # 'horizontal_flip'=True,
    # 'vertical_flip'=True,
}

generator = ImageDataGenerator(**generator_args)

flow_args = {
    # 'directory': '../dataset_v2/train/classes',
    'target_size':(128, 128), #(200, 200) for ResNet50
    'batch_size':32
}

train_gen = generator.flow_from_directory(directory='../dataset_v2/train/classes', **flow_args)
valid_gen = generator.flow_from_directory(directory='../dataset_v2/train/cropped', **flow_args, shuffle=False)


from collections import Counter
Nm = np.array(sorted(Counter(train_gen.classes).items()))[:,1]
w0 = np.maximum(Nm.astype(np.float)/train_gen.samples, 0.1)
w = (1 - w0) / w0

def WeightedCELoss(ys, ts):
    return 1/(Nm.sum() + len(w)) * (w * ys * tf.log(ts) + (1 - ys) * tf.log(1 - ts))

compile_args = {
    'optimizer':optimizers.SGD(lr=0.01, momentum=0.9), # 0.002
    'loss': losses.binary_crossentropy, #WeightedCELoss
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

model = MNet()
model.summary()
model.compile(**compile_args)
model.fit_generator(**fit_gen_args)
model.save('mnet100.h5')

model = models.load_model('mnet100.h5')

# Get predictions
predictions_proba = model.predict_generator(valid_gen, steps=len(valid_gen))
predictions = predictions_proba > 0.1

np.save('predictions.npy', predictions)
predictions = np.load('predictions.npy')

# Get targets
train = pd.read_csv('../dataset_v2/train.csv')
train = train.replace(' ', '_', regex=True)
train = train.replace('/', '_', regex=True)
train = train.drop(['p1_x', 'p_1y', ' p2_x', ' p2_y', ' p3_x', ' p3_y', ' p4_x', ' p4_y'], axis=1)
for category in ['general_class', 'sub_class', 'color']:
    train[category] = train[category].astype('category')
train = pd.get_dummies(train, prefix='', prefix_sep='')
train = train.groupby(['image_id', 'tag_id']).first()
train = train.reindex_axis(sorted(train.columns), axis=1)
targets = train.values

def APScore(ys, ts):    
    K = 1 #= (ts==1).sum()
    TP = ys[ts==1].sum()
    FP = ys[ts!=1].sum()
    return (1/K) * (TP / (TP + FP))

assert(predictions.shape == targets.shape)
N_c = targets.shape[1]
AP = np.zeros(N_c)
for category in range(N_c):
    AP[category] = APScore(predictions[:,category], targets[:,category])

MAP = np.mean(AP)

(MAP, AP)
