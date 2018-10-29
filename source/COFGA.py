# https://arxiv.org/pdf/1808.09001.pdf
# https://www.learnopencv.com/installing-deep-learning-frameworks-on-ubuntu-with-cuda-support/
# https://www.learnopencv.com/keras-tutorial-fine-tuning-using-pre-trained-models/

import numpy as np
import matplotlib.pyplot as plt

from keras import models, layers, optimizers, losses, applications
from keras.preprocessing.image import ImageDataGenerator

# def AP(ys, ts):
#     """
#         Average precision by category
#     """
#     predictions = max(ys, 1)
#     precision = eq(predictions, ts)
#     return mean(precision.float())

# def MAP(aps):
#     """
#         Mean average precision across all categories
#     """
#     return mean(aps)


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
        for layer in base.layers[:-4]:
            layer.trainable = False

        self.add(base)
        self.add(layers.Flatten())
        self.add(layers.Dense(sizes[-2], activation='relu'))
        self.add(layers.Dense(sizes[-1], activation='sigmoid'))
mnet = MNet()
mnet.summary()

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
        for layer in base.layers[:-4]:
            layer.trainable = False

        self.add(base)
        self.add(layers.Flatten())
        self.add(layers.Dense(sizes[-2], activation='relu'))
        self.add(layers.Dense(sizes[-1], activation='sigmoid'))
rnet = RNet()
rnet.summary()
    
# https://keras.io/preprocessing/image/#imagedatagenerator-class
# Data augmentation
# train_datagen = ImageDataGenerator(
#     rescale=1./255,
#     rotation_range=20,
#     width_shift_range=0.2,
#     height_shift_range=0.2,
#     horizontal_flip=True,
#     vertical_flip=True,
#     fill_mode='nearest')
 
# validation_datagen = ImageDataGenerator(rescale=1./255)

generator_args = {
    'train_dir':'./dataset_v2/train', 
    'val_dir':'./dataset_v2/test', 
    'train_batch_size':100, 
    'val_batch_size':10, 
    'image_size':128
}

def get_generators(train_dir, val_dir, 
        train_batch_size, val_batch_size, image_size):
    train = ImageDataGenerator().flow_from_directory(train_dir,
            target_size=(image_size, image_size),
            batch_size=train_batch_size)    
    valid = ImageDataGenerator().flow_from_directory(val_dir,
            target_size=(image_size, image_size),
            batch_size=val_batch_size, shuffle=False)
    return train, valid

train_gen, valid_gen = get_generators(**generator_args)

train_args = {
    'train':train_gen,
    'valid':valid_gen,
    'epochs':100, 
    'loss':losses.binary_crossentropy,
    'optimizer':optimizers.SGD(lr=0.01, momentum=0.9)
}

def train(model, train, valid, epochs, loss, optimizer):
    # Compile the model
    model.compile(optimizer, loss, metrics=['acc'])
    # Train the model
    model.fit_generator(train, epochs=epochs,
        steps_per_epoch=train.samples/train.batch_size,
        validation_steps=valid.samples/valid.batch_size,
        validation_data=valid, 
        verbose=1)
 
    # # Save the model
    # model.save('small_last4.h5')

train(mnet, **train_args)
mnet.evaluate()

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