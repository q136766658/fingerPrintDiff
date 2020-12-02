#Import necessary libraries
from keras import layers
from keras import models
from keras.models import Model
from keras.optimizers import Adam, Nadam, SGD
from keras.callbacks import TensorBoard, Callback
import tensorflow as tf
import keras
# def create_model_2():
#         model = models.Sequential()
#         model.add(layers.Conv2D(32, (5, 5),padding='same', activation='relu',input_shape=(96, 96, 1),
#                                 kernel_regularizer=keras.regularizers.l2(0.01)))
#         model.add(layers.Conv2D(32, (3, 3),padding='same', activation='relu',
#                                 kernel_regularizer=keras.regularizers.l2(0.01)))
#         model.add(layers.MaxPooling2D((2, 2)))
#         model.add(layers.Flatten())
#         model.add(layers.Dense(128, activation='relu'))
#         model.add(layers.Dropout(0.5))
#         model.add(layers.Dense(1, activation='sigmoid'))
#         return model

def create_model_2():
        model = models.Sequential()
        model.add(layers.Conv2D(32, (3, 3),padding='same', activation='relu',input_shape=(96, 96, 1),
                                kernel_regularizer=keras.regularizers.l2(0.01)))
        model.add(layers.MaxPooling2D((2, 2)))
        model.add(layers.Flatten())
        model.add(layers.Dense(32, activation='relu'))
        # model.add(layers.Dropout(0.5))
        model.add(layers.Dense(1, activation='sigmoid'))
        return model

def create_model_5():
        model = models.Sequential()
        model.add(layers.Conv2D(32, (3, 3),padding='same', activation='relu',input_shape=(96, 96, 1),
                                kernel_regularizer=keras.regularizers.l2(0.01)))
        model.add(layers.MaxPooling2D((2, 2)))
        model.add(layers.Flatten())
        model.add(layers.Dense(64, activation='relu',kernel_regularizer=keras.regularizers.l2(0.01)))
        model.add(layers.Dense(5, activation='softmax'))
        return model