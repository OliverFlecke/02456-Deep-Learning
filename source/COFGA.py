# https://arxiv.org/pdf/1808.09001.pdf
# https://www.learnopencv.com/installing-deep-learning-frameworks-on-ubuntu-with-cuda-support/
# https://www.learnopencv.com/keras-tutorial-fine-tuning-using-pre-trained-models/
# https://keras.io/preprocessing/image/#imagedatagenerator-class

import os
import numpy as np
import pandas as pd
import matplotlib.pyplot as plt

from collections import Counter
from keras import backend as K
from keras import models, layers, optimizers, losses, applications
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


generator_args = {
    # 'rescale':1./255,
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
    # 'save_to_dir':'../dataset_v2/train/saved'
}

train_gen = generator.flow_from_directory(directory='../dataset_v2/train/classes', **flow_args)
# valid_gen = generator.flow_from_directory(directory='../dataset_v2/train/cropped', **flow_args)
test_gen = generator.flow_from_directory(directory='../dataset_v2/test/', **flow_args, shuffle=False)

def get_loss_weights(generator):
    Nm = np.array(sorted(Counter(generator.classes).items()))[:,1]
    w0 = np.maximum(Nm.astype(np.float)/generator.samples, 0.1)
    return (1 - w0) / w0

compile_args = {
    'optimizer':optimizers.SGD(lr=0.01, momentum=0.9), # 0.01 / 0.002
    'loss': losses.binary_crossentropy,
    # 'loss_weights':get_loss_weights(train_gen),
    'metrics':['acc']
}
fit_gen_args = {
    'generator':train_gen,
    # 'validation_data':valid_gen,
    'steps_per_epoch':len(train_gen),
    # 'validation_steps':len(valid_gen),
    'epochs':50,
}

# model = MNet(len(train_gen.class_indices), 'softmax')
model = MNet()
model.summary()
model.compile(**compile_args)
model.fit_generator(**fit_gen_args)
model.save('../models/mnet50')