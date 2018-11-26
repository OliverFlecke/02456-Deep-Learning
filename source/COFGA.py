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
from keras import models, layers, optimizers, losses, applications, callbacks
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
        'validation_split':0.3,
        'rotation_range':45,
        'width_shift_range':0.2,
        'height_shift_range':0.2,
        'horizontal_flip':True,
        'vertical_flip':True,
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
        
        outputs = len(train_gen.class_indices)
        samples = train_gen.samples + valid_gen.samples
        epochs = int(1e4 / np.sqrt(samples))
        lr = np.sqrt(samples) / 1e4
        patience = epochs // 10

        model_args = {
            'outputs':outputs,
            'output_layer':'softmax'
        }
    
        compile_args = {
            'optimizer':optimizers.SGD(
                lr=lr, 
                momentum=0.9, 
                decay=0.1),
            'loss':losses.binary_crossentropy,
            'metrics':['acc']
        }

        fit_gen_args = {
            'generator':train_gen,
            'validation_data':valid_gen,
            'steps_per_epoch':len(train_gen),
            'validation_steps':len(valid_gen),
            'epochs':epochs,
            # 'class_weight':get_loss_weights(train_gen),
            'callbacks':[callbacks.EarlyStopping(
                monitor='val_acc', 
                min_delta=1e-4, 
                patience=patience, 
                restore_best_weights=True)],
        }

        model = MNet(**model_args)    
        model.compile(**compile_args)
        history[category] = model.fit_generator(**fit_gen_args)
        model.save_weights(f'{target_dir}{category}.h5')

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
    return dict(enumerate( (1 - w0) / w0 ))

source_dir = '../dataset_v2/train/classes/'
target_dir = '../models/classes/'
category = ''
categories = os.listdir(source_dir)
# history = train_classifier(categories, source_dir, target_dir)
# plot_history(history)

generator_args = {
    # 'validation_split':0.3,
    'rotation_range':45,
    'width_shift_range':0.2,
    'height_shift_range':0.2,
    'horizontal_flip':True,
    'vertical_flip':True,
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

test_gen = generator.flow_from_directory(
    directory='../dataset_v2/test/', 
    shuffle=False, 
    target_size=(128, 128), 
    batch_size=32
)

outputs = len(train_gen.class_indices)
samples = train_gen.samples + valid_gen.samples
epochs = int(1e4 / np.sqrt(samples))
lr = np.sqrt(samples) / 1e4
patience = epochs // 10

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
    # 'validation_data':valid_gen,
    'steps_per_epoch':len(train_gen),
    # 'validation_steps':len(valid_gen),
    'epochs':50,
    # 'class_weight':get_class_weight(train_gen),
    # 'callbacks':[callbacks.EarlyStopping(
    #     monitor='val_acc', 
    #     min_delta=1e-4, 
    #     patience=patience, 
    #     restore_best_weights=True)],
}

model = MNet()    
model.compile(**compile_args)
model.fit_generator(**fit_gen_args)
model.save_weights('../models/mnet50aug.h5')

# proba = model.predict_generator(test_gen, steps=len(test_gen))
# cols = ['large vehicle', 'small vehicle']

