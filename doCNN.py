from sklearn.metrics import plot_confusion_matrix
import tensorflow as tf
from tensorflow.keras.layers import Input, Dense, Dropout, Flatten, Conv2D, MaxPool2D
from tensorflow.keras import layers 
import numpy as np
import scipy
import matplotlib
import matplotlib.pyplot as plt
from tensorflow.keras.utils import image_dataset_from_directory, plot_model
from tensorflow.keras.models import Sequential
from pathlib import Path

np.random.seed(0)
tf.random.set_seed(0)



batch_size = 32
img_height = 150
img_width = 150

class doCNN:
    img_height = 150
    img_width = 150 

    def getCNNData(dataset = "train",subset = "training"):
        dset = "seg_"+dataset
        rootPath = Path.cwd()
        data_dir = rootPath / "archive" / "seg_train" /dset
        ds = image_dataset_from_directory(
        data_dir,
        validation_split=0.2,
        subset=subset,
        seed=123,
        labels='inferred',
        seed=123,
        image_size=(150, 150),
        batch_size=32)
        # AUTOTUNE = tf.data.AUTOTUNE
        # ds = ds.cache().shuffle(1000).prefetch(buffer_size=AUTOTUNE)
        return ds

    def doCNN(trainds,valds,activation='relu',epochs = 10):
        """
        build model, compile model
        """
        num_classes = len(trainds.class_names)
        model = Sequential([
        layers.Rescaling(1./255, input_shape=(img_height, img_width, 3)),
        layers.Conv2D(16, 3, padding='same', activation=activation),
        layers.MaxPooling2D(),
        layers.Conv2D(32, 3, padding='same', activation=activation),
        layers.MaxPooling2D(),
        layers.Conv2D(64, 3, padding='same', activation=activation),
        layers.MaxPooling2D(),
        layers.Flatten(),
        layers.Dense(128, activation='relu'),
        layers.Dense(num_classes)
        ])
        model.compile(optimizer='adam',
              loss=tf.keras.losses.SparseCategoricalCrossentropy(from_logits=True),
              metrics=['accuracy'])
        history = model.fit(trainds,validation_data =valds,epochs=epochs)

        return history,model

    def plots(history):
        #TODO: accuracy and loss plots 
        """
        Plot accuracy, and loss plots
        """
        pass
    
    def testCNN(testData):
        pass

    def validateCNN(validationData):
        pass



        




    #def doCNN(ds):



if __name__ == "__main__":
    train_ds = doCNN.getCNNData()
    val_ds = doCNN.getCNNData(dataset="train",subset="validation")
    h,model  = doCNN.doCNN(train_ds)
    plot_model(model,to_file='cnn_model.png',show_shapes = True,show_layer_activations=True)


    


