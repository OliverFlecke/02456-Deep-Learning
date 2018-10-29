# https://arxiv.org/pdf/1808.09001.pdf
# https://www.learnopencv.com/installing-deep-learning-frameworks-on-ubuntu-with-cuda-support/
# https://www.learnopencv.com/keras-tutorial-fine-tuning-using-pre-trained-models/

import numpy as np
import matplotlib.pyplot as plt

from keras import models, layers, optimizers, applications

# Auxilliary functions

def freeze_layers(model, trainable):
    """
        Freeze all layers except the last *trainable* ones.
    """
    for layer in model.layers[:-trainable]:
        layer.trainable = False

# Loss functions

def AP(ys, ts):
    """
        Average precision by category
    """
    predictions = max(ys, 1)
    precision = eq(predictions, ts)
    return mean(precision.float())

def MAP(aps):
    """
        Mean average precision across all categories
    """
    return mean(aps)


# Models

class MNet(models.Sequential):
    """
        Using pre-trained MobileNet without head
    """
    def __init__(self):
        super(MNet, self).__init__()
        
        inputs = [37, 1024, (4,4,1024), (128,128,3)]

        self.add(layers.Dense(inputs[0], activation='sigmoid'))
        self.add(layers.Dense(inputs[1], activation='relu'))
        self.add(layers.Reshape(inputs[2]))
        self.add(applications.MobileNet(weights='imagenet', include_top=False))
        freeze_layers(self.layers[-1], 4)
        self.add(layers.Reshape(inputs[3]))


class RNet(models.Sequential):
    """
        Using pre-trained ResNet50 without head
    """
    def __init__(self):
        super(RNet, self).__init__()
        
        inputs = [37, 1024, (7,7,1024), (200,200,3)]

        self.add(layers.Dense(inputs[0], activation='sigmoid'))
        self.add(layers.Dense(inputs[1], activation='relu'))
        self.add(layers.Reshape(inputs[2]))
        self.add(applications.ResNet50(weights='imagenet', include_top=False))
        freeze_layers(self.layers[-1], 4)
        self.add(layers.Reshape(inputs[3]))

    
lr = 0.01
momentum = 0.9
optimizer = optimizers.SGD(lr, momentum)

batch_size = 32
epochs = 100


def train():
    # Compile the model
    model.compile(loss='categorical_crossentropy',
                optimizer=optimizer,
                metrics=['acc'])
    # Train the model
    history = model.fit_generator(
        train_generator,
        steps_per_epoch=train_generator.samples/train_generator.batch_size ,
        epochs=epochs,
        validation_data=validation_generator,
        validation_steps=validation_generator.samples/validation_generator.batch_size,
        verbose=1)
 
    # # Save the model
    # model.save('small_last4.h5')

def eval():
    """
        Check performance by visualizing loss and accuracy curves
    """
    acc = history.history['acc']
    val_acc = history.history['val_acc']
    loss = history.history['loss']
    val_loss = history.history['val_loss']
    
    epochs = range(len(acc))
    
    plt.plot(epochs, acc, 'b', label='Training acc')
    plt.plot(epochs, val_acc, 'r', label='Validation acc')
    plt.title('Training and validation accuracy')
    plt.legend()
    
    plt.figure()
    
    plt.plot(epochs, loss, 'b', label='Training loss')
    plt.plot(epochs, val_loss, 'r', label='Validation loss')
    plt.title('Training and validation loss')
    plt.legend()
    
    plt.show()

def visualize():
    """
        Visualize the errors that occurred
    """
    # Create a generator for prediction
    validation_generator = validation_datagen.flow_from_directory(
            validation_dir,
            target_size=(image_size, image_size),
            batch_size=val_batchsize,
            class_mode='categorical',
            shuffle=False)
    
    # Get the filenames from the generator
    fnames = validation_generator.filenames
    
    # Get the ground truth from generator
    ground_truth = validation_generator.classes
    
    # Get the label to class mapping from the generator
    label2index = validation_generator.class_indices
    
    # Getting the mapping from class index to class label
    idx2label = dict((v,k) for k,v in label2index.items())
    
    # Get the predictions from the model using the generator
    predictions = model.predict_generator(validation_generator, steps=validation_generator.samples/validation_generator.batch_size,verbose=1)
    predicted_classes = np.argmax(predictions,axis=1)
    
    errors = np.where(predicted_classes != ground_truth)[0]
    print("No of errors = {}/{}".format(len(errors),validation_generator.samples))
    
    # Show the errors
    for i in range(len(errors)):
        pred_class = np.argmax(predictions[errors[i]])
        pred_label = idx2label[pred_class]
        
        title = 'Original label:{}, Prediction :{}, confidence : {:.3f}'.format(
            fnames[errors[i]].split('/')[0],
            pred_label,
            predictions[errors[i]][pred_class])
        
        original = load_img('{}/{}'.format(validation_dir,fnames[errors[i]]))
        plt.figure(figsize=[7,7])
        plt.axis('off')
        plt.title(title)
        plt.imshow(original)
        plt.show()