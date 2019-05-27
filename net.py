import tensorflow as tf
import keras
from tensorflow.keras.layers import Conv3D, Conv3DTranspose, Flatten, Dense, MaxPooling3D, UpSampling3D
import numpy as np
from tensorflow.keras import optimizers
from tensorflow.keras import backend as K

def weighted_categorical_crossentropy(weights):
    """
    A weighted version of keras.objectives.categorical_crossentropy
    
    Variables:
        weights: numpy array of shape (C,) where C is the number of classes
    
    Usage:
        weights = np.array([0.5,2,10]) # Class one at 0.5, class 2 twice the normal weights, class 3 10x.
        loss = weighted_categorical_crossentropy(weights)
        model.compile(loss=loss,optimizer='adam')
    """
    
    weights = K.variable(weights)
        
    def loss(y_true, y_pred):
        # scale predictions so that the class probas of each sample sum to 1
        y_pred /= K.sum(y_pred, axis=-1, keepdims=True)
        # clip to prevent NaN's and Inf's
        y_pred = K.clip(y_pred, K.epsilon(), 1 - K.epsilon())
        # calc
        loss = y_true * K.log(y_pred) * weights
        loss = -K.sum(loss, -1)
        return loss
    
    return loss

def weighted_mean_squared(weights):
    """
    A weighted version of keras.objectives.categorical_crossentropy
    
    Variables:
        weights: numpy array of shape (C,) where C is the number of classes
    
    Usage:
        weights = np.array([0.5,2,10]) # Class one at 0.5, class 2 twice the normal weights, class 3 10x.
        loss = weighted_categorical_crossentropy(weights)
        model.compile(loss=loss,optimizer='adam')
    """
    
    weights = K.variable(weights)
        
    def loss(y_true, y_pred):
        # scale predictions so that the class probas of each sample sum to 1
        y_pred /= K.sum(y_pred, axis=-1, keepdims=True)
        # clip to prevent NaN's and Inf's
        y_pred = K.clip(y_pred, K.epsilon(), 1 - K.epsilon())
        # calc
        loss = y_true * K.log(y_pred) * weights
        loss = -K.sum(loss, -1)
        return loss
    
    return loss


def conv_model(loss = "mean_squared_error"):
    model = tf.keras.models.Sequential()
    model.add(Conv3D(filters=10,kernel_size=3,strides=(1,1,1),padding="same", activation="relu"))
    model.add(Conv3D(filters=10,kernel_size=3,strides=(1,1,1),padding="same", activation="relu"))
    model.add(Conv3D(filters=1,kernel_size=3,strides=(1,1,1),padding="same", activation="sigmoid"))
    #model.compile(loss=loss_fn_w_xe_logits(1), optimizer="sgd", metrics=["acc"])
    sgd = optimizers.SGD(lr=0.01, decay=1e-6, momentum=0.9, nesterov=True)
    model.compile(loss=loss, optimizer=sgd, metrics=["acc"])
    return model
    #model.compile(optimizer='adadelta', loss='categorical_crossentropy',
    #          metrics=['acc'])
    #return model

def conv_model_simple(loss = "mean_squared_error"):
    model = tf.keras.models.Sequential()
    model.add(Conv3D(filters=1,kernel_size=3,strides=(1,1,1),padding="same", activation="sigmoid"))
    sgd = optimizers.SGD(lr=0.01, decay=1e-6, momentum=0.9, nesterov=True)
    model.compile(loss=loss, optimizer=sgd, metrics=["acc"])
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

def conv_model_auto(loss = 'mean_squared_error'):
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
              loss=loss,
              metrics=['accuracy'])
    return model

def conv_model_auto_simple(loss = 'mean_squared_error'):
    model = tf.keras.models.Sequential()
    
    model.add(Conv3D(filters=30,kernel_size=3,strides=(1,1,1),padding="same", activation="relu"))
    
    model.add(MaxPooling3D(pool_size=(2, 2, 2), strides=None, padding='valid'))
    
    model.add(Conv3D(filters=30,kernel_size=3,strides=(1,1,1),padding="same", activation="relu"))
    
    
    model.add(UpSampling3D((2,2,2)))
    
    model.add(Conv3D(filters=30,kernel_size=3,strides=(1,1,1),padding="same", activation="relu"))
    model.add(Conv3D(filters=1,kernel_size=3,strides=(1,1,1),padding="same", activation="sigmoid"))
    
    sgd = optimizers.SGD(lr=0.01, decay=1e-6, momentum=0.9, nesterov=True)
    model.compile(loss=loss, optimizer=sgd, metrics=["accuracy"])
    #model.compile(optimizer='sgd',
   #           loss='categorical_crossentropy',
    #          metrics=['accuracy'])
    return model

def conv_model_auto_deep(loss = 'mean_squared_error'):
    model = tf.keras.models.Sequential()
    
    model.add(Conv3D(filters=30,kernel_size=3,strides=(1,1,1),padding="same", activation="relu"))
    model.add(Conv3D(filters=30,kernel_size=3,strides=(1,1,1),padding="same", activation="relu"))
    model.add(Conv3D(filters=30,kernel_size=3,strides=(1,1,1),padding="same", activation="relu"))
    
    model.add(MaxPooling3D(pool_size=(2, 2, 2), strides=None, padding='valid'))
    
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
    model.add(Conv3D(filters=30,kernel_size=3,strides=(1,1,1),padding="same", activation="relu"))
    
    model.add(UpSampling3D((2,2,2)))
    
    model.add(Conv3D(filters=30,kernel_size=3,strides=(1,1,1),padding="same", activation="relu"))
    model.add(Conv3D(filters=30,kernel_size=3,strides=(1,1,1),padding="same", activation="relu"))
    model.add(Conv3D(filters=1,kernel_size=3,strides=(1,1,1),padding="same", activation="sigmoid"))
    
    model.compile(optimizer='sgd',
              loss=loss,
              metrics=['accuracy'])
    return model

models = [conv_model, conv_model_deep, conv_model_auto()]