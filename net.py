import tensorflow as tf
import keras
from tensorflow.keras.layers import Conv3D, Conv3DTranspose, Flatten, Dense
import numpy as np

def conv_auto_model():
    model = tf.keras.models.Sequential()
    model.add(Conv3D(filters=10,kernel_size=3,strides=(1,1,1),padding="same", activation="relu"))
    model.add(Conv3D(filters=10,kernel_size=3,strides=(1,1,1),padding="same", activation="relu"))
    model.add(Conv3D(filters=10,kernel_size=3,strides=(1,1,1),padding="same", activation="relu"))
    model.compile(optimizer='adam',
              loss='categorical_crossentropy',
              metrics=['accuracy'])
    return model

