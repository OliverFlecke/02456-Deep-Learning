# https://arxiv.org/pdf/1808.09001.pdf
# https://www.learnopencv.com/installing-deep-learning-frameworks-on-ubuntu-with-cuda-support/
# https://www.learnopencv.com/keras-tutorial-fine-tuning-using-pre-trained-models/

import os
import numpy as np
import matplotlib.pyplot as plt

import tensorflow as tf
from keras import models, layers, optimizers, losses, applications
from keras.preprocessing.image import ImageDataGenerator

def MAP(model, generator):
    """
        Mean average precision across all categories
    """
    predictions = model.predict_generator(generator, steps=len(generator))
    predictions = np.argmax(predictions, axis=-1)
    targets = generator.classes

    AP = np.zeros(len(generator.class_indices))
    
    for c in np.unique(generator.classes):
    # for c in generator.class_indices.values():
        AP[c] = np.mean(predictions[targets==c] == c)
    
    # AP[np.isnan(AP)] = 0 # or 1?

    return np.mean(AP), AP

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
    'validation_split':0.3,
    # 'rotation_range'=45,
    # 'width_shift_range'=0.2,
    # 'height_shift_range'=0.2,
    # 'horizontal_flip'=True,
    # 'vertical_flip'=True,
}

generator = ImageDataGenerator(**generator_args)

flow_args = {
    'directory': os.path.join(os.path.dirname(os.path.realpath(__file__)), '../dataset_v2/train/classes'),
    'target_size':(128, 128), #(200, 200) for ResNet50
    'batch_size':32
}

train_gen = generator.flow_from_directory(subset='training', **flow_args)
valid_gen = generator.flow_from_directory(subset='validation', **flow_args)


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
    'validation_data':valid_gen,
    'steps_per_epoch':len(train_gen),
    'validation_steps':len(valid_gen),
    'epochs':1, 
    'class_weight':None
}

model = RNet()
model.summary()
model.compile(**compile_args)
model.fit_generator(**fit_gen_args)
MAP(model, valid_gen)

# Get NaNs indices
# indices = np.array(range(37))
# indices[np.isnan(AP)]

# Convert to labels
# label_map = (train_gen.class_indices)
# label_map = dict((v,k) for k,v in label_map.items()) #flip k,v
# predictions = [label_map[k] for k in predictions]

# mnet.save('mnet.h5')

# def eval():
#     """
#         Check performance by visualizing loss and accuracy curves
#     """
#     acc = history.history['acc']
#     val_acc = history.history['val_acc']
#     loss = history.history['loss']
#     val_loss = history.history['val_loss']
    
#     epochs = range(len(acc))
    
#     plt.plot(epochs, acc, 'b', label='Training acc')
#     plt.plot(epochs, val_acc, 'r', label='Validation acc')
#     plt.title('Training and validation accuracy')
#     plt.legend()
    
#     plt.figure()
    
#     plt.plot(epochs, loss, 'b', label='Training loss')
#     plt.plot(epochs, val_loss, 'r', label='Validation loss')
#     plt.title('Training and validation loss')
#     plt.legend()
    
#     plt.show()

# def visualize():
#     """
#         Visualize the errors that occurred
#     """
#     # Create a generator for prediction
#     validation_generator = validation_datagen.flow_from_directory(
#             validation_dir,
#             target_size=(image_size, image_size),
#             batch_size=val_batchsize,
#             class_mode='categorical',
#             shuffle=False)
    
#     # Get the filenames from the generator
#     fnames = validation_generator.filenames
    
#     # Get the ground truth from generator
#     ground_truth = validation_generator.classes
    
#     # Get the label to class mapping from the generator
#     label2index = validation_generator.class_indices
    
#     # Getting the mapping from class index to class label
#     idx2label = dict((v,k) for k,v in label2index.items())
    
#     # Get the predictions from the model using the generator
#     predictions = model.predict_generator(validation_generator, steps=validation_generator.samples/validation_generator.batch_size,verbose=1)
#     predicted_classes = np.argmax(predictions,axis=1)
    
#     errors = np.where(predicted_classes != ground_truth)[0]
#     print("No of errors = {}/{}".format(len(errors),validation_generator.samples))
    
#     # Show the errors
#     for i in range(len(errors)):
#         pred_class = np.argmax(predictions[errors[i]])
#         pred_label = idx2label[pred_class]
        
#         title = 'Original label:{}, Prediction :{}, confidence : {:.3f}'.format(
#             fnames[errors[i]].split('/')[0],
#             pred_label,
#             predictions[errors[i]][pred_class])
        
#         original = load_img('{}/{}'.format(validation_dir,fnames[errors[i]]))
#         plt.figure(figsize=[7,7])
#         plt.axis('off')
#         plt.title(title)
#         plt.imshow(original)
#         plt.show()