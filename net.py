import tensorflow as tf
import keras
from tensorflow.keras.layers import Conv3D, Conv3DTranspose, Flatten, Dense, MaxPooling3D, UpSampling3D
import numpy as np
from tensorflow.keras import optimizers


def conv_model():
    model = tf.keras.models.Sequential()
    model.add(Conv3D(filters=10,kernel_size=3,strides=(1,1,1),padding="same", activation="relu"))
    model.add(Conv3D(filters=10,kernel_size=3,strides=(1,1,1),padding="same", activation="relu"))
    model.add(Conv3D(filters=1,kernel_size=3,strides=(1,1,1),padding="same", activation="sigmoid"))
    #model.compile(loss=loss_fn_w_xe_logits(1), optimizer="sgd", metrics=["acc"])
    sgd = optimizers.SGD(lr=0.01, decay=1e-6, momentum=0.9, nesterov=True)
    model.compile(loss='mean_squared_error', optimizer=sgd, metrics=["acc"])
    return model
    #model.compile(optimizer='adadelta', loss='categorical_crossentropy',
    #          metrics=['acc'])
    #return model

def conv_model_simple():
    model = tf.keras.models.Sequential()
    model.add(Conv3D(filters=1,kernel_size=3,strides=(1,1,1),padding="same", activation="sigmoid"))
    sgd = optimizers.SGD(lr=0.01, decay=1e-6, momentum=0.9, nesterov=True)
    model.compile(loss='mean_squared_error', optimizer=sgd, metrics=["acc"])
    return model

def conv_model_deep():
    model = tf.keras.models.Sequential()
    
    model.add(Conv3D(filters=30,kernel_size=3,strides=(1,1,1),padding="same", activation="relu"))
    model.add(Conv3D(filters=30,kernel_size=3,strides=(1,1,1),padding="same", activation="relu"))
    model.add(Conv3D(filters=30,kernel_size=3,strides=(1,1,1),padding="same", activation="relu"))
    
    model.add(Conv3D(filters=30,kernel_size=3,strides=(1,1,1),padding="same", activation="relu"))
    model.add(Conv3D(filters=30,kernel_size=3,strides=(1,1,1),padding="same", activation="relu"))
    model.add(Conv3D(filters=30,kernel_size=3,strides=(1,1,1),padding="same", activation="relu"))
    
    model.add(Conv3D(filters=30,kernel_size=3,strides=(1,1,1),padding="same", activation="relu"))
    model.add(Conv3D(filters=30,kernel_size=3,strides=(1,1,1),padding="same", activation="relu"))
    model.add(Conv3D(filters=1,kernel_size=3,strides=(1,1,1),padding="same", activation="sigmoid"))
    
    model.compile(optimizer='adam',
              loss='categorical_crossentropy',
              metrics=['accuracy'])
    return model

def conv_model_auto():
    model = tf.keras.models.Sequential()
    
    model.add(Conv3D(filters=30,kernel_size=3,strides=(1,1,1),padding="same", activation="relu"))
    model.add(Conv3D(filters=30,kernel_size=3,strides=(1,1,1),padding="same", activation="relu"))
    model.add(Conv3D(filters=30,kernel_size=3,strides=(1,1,1),padding="same", activation="relu"))
    
    model.add(MaxPooling3D(pool_size=(2, 2, 2), strides=None, padding='valid'))
    
    model.add(Conv3D(filters=30,kernel_size=3,strides=(1,1,1),padding="same", activation="relu"))
    model.add(Conv3D(filters=30,kernel_size=3,strides=(1,1,1),padding="same", activation="relu"))
    model.add(Conv3D(filters=30,kernel_size=3,strides=(1,1,1),padding="same", activation="relu"))
    
    model.add(UpSampling3D((2,2,2)))
    
    model.add(Conv3D(filters=30,kernel_size=3,strides=(1,1,1),padding="same", activation="relu"))
    model.add(Conv3D(filters=30,kernel_size=3,strides=(1,1,1),padding="same", activation="relu"))
    model.add(Conv3D(filters=1,kernel_size=3,strides=(1,1,1),padding="same", activation="sigmoid"))
    
    model.compile(optimizer='sgd',
              loss='categorical_crossentropy',
              metrics=['accuracy'])
    return model



models = [conv_model, conv_model_deep, conv_model_auto()]