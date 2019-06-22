import tensorflow as tf
from tensorflow import keras
from tensorflow.keras import metrics
from tensorflow.keras.layers import Conv3D, Conv3DTranspose, Flatten, Dense, Dropout, MaxPooling3D, UpSampling3D, \
    concatenate, Add, Activation, SpatialDropout3D
import numpy as np
from tensorflow.keras import optimizers
from tensorflow.keras import backend as K
from tensorflow.keras.models import Model
import os
from custom_layers import norm
import config


def dice_coef(y_true, y_pred):
    smooth = 1.
    y_true_f = tf.keras.layers.Flatten()(y_true)
    y_pred_f = tf.keras.layers.Flatten()(y_pred)
    intersection = tf.reduce_sum(y_true_f * y_pred_f)
    return (2. * intersection + smooth) / (tf.reduce_sum(y_true_f) + tf.reduce_sum(y_pred_f) + smooth)


def dice_coef_loss(y_true, y_pred):
    return 1.0 - dice_coef(y_true, y_pred)


def weighted_crossentropy(beta):
    def convert_to_logits(y_pred):
        # see https://github.com/tensorflow/tensorflow/blob/r1.10/tensorflow/python/keras/backend.py#L3525
        y_pred = tf.clip_by_value(y_pred, tf.keras.backend.epsilon(), 1 - tf.keras.backend.epsilon())

        return tf.math.log(y_pred / (1 - y_pred))

    def loss(y_true, y_pred):
        y_pred = convert_to_logits(y_pred)
        loss = tf.nn.weighted_cross_entropy_with_logits(logits=y_pred, targets=y_true, pos_weight=beta)

        return tf.reduce_mean(loss)

    return loss


def focal_loss(alpha=0.25, gamma=2):
    def focal_loss_with_logits(logits, targets, alpha, gamma, y_pred):
        weight_a = alpha * (1 - y_pred) ** gamma * targets
        weight_b = (1 - alpha) * y_pred ** gamma * (1 - targets)

        return (tf.log1p(tf.exp(-tf.abs(logits))) + tf.nn.relu(-logits)) * (weight_a + weight_b) + logits * weight_b

    def loss(y_true, y_pred):
        y_pred = tf.clip_by_value(y_pred, tf.keras.backend.epsilon(), 1 - tf.keras.backend.epsilon())
        logits = tf.log(y_pred / (1 - y_pred))

        loss = focal_loss_with_logits(logits=logits, targets=y_true, alpha=alpha, gamma=gamma, y_pred=y_pred)

        return tf.reduce_mean(loss)

    return loss


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


class UNet:
    def __init__(self, conv_count=2, filter_count=40, level_count=2, dropout=True, optimizer="adam",
                 loss='mean_squared_error', output_dim=1, residual=True, concatenate=True):
        self.conv_count = conv_count
        self.filter_count = filter_count
        self.level_count = level_count
        self.dropout = dropout
        self.loss = loss
        self.output_dim = output_dim
        self.residual = residual
        self.concatenate = concatenate
        self.lr = 0
        if optimizer == "adam":
            self.lr = 0.0001
            self.optimizer = keras.optimizers.Adam(lr=self.lr)

        elif optimizer == "sgd":
            self.lr = 0.01
            self.optimizer = keras.optimizers.SGD(lr=self.lr, decay=1e-3, momentum=0.9, nesterov=True)
        else:
            self.optimizer = optimizer

    def residual_section(self, x, filter_count):
        branch = self.conv_section(x, filter_count)
        return Add()([x, branch])

    def conv_section(self, x, filter_count):
        # TODO norm
        for i in range(self.conv_count):
            x = norm.GroupNormalization(groups=filter_count // 2)(x)
            x = Activation("relu")(x)
            x = Conv3D(filters=filter_count, kernel_size=3, strides=(1, 1, 1), padding="same")(x)

        if self.dropout:
            x = SpatialDropout3D(0.2)(x)
        return x

    def residual_end_section(self, x, filter_count):

        y = Conv3D(filters=filter_count, kernel_size=3, strides=(1, 1, 1), padding="same", activation="relu")(x)
        y = Add()([y, x])
        y = Conv3D(filters=filter_count, kernel_size=1, strides=(1, 1, 1), padding="same", activation="relu")(y)

        Output = Conv3D(filters=self.output_dim, kernel_size=1, strides=(1, 1, 1), padding="same",
                        activation="sigmoid")(y)
        return Output

    def conv_end_section(self, y, filter_count):
        y = Conv3D(filters=filter_count, kernel_size=3, strides=(1, 1, 1), padding="same", activation="relu")(y)
        y = Conv3D(filters=filter_count, kernel_size=1, strides=(1, 1, 1), padding="same", activation="relu")(y)
        Output = Conv3D(filters=self.output_dim, kernel_size=1, strides=(1, 1, 1), padding="same",
                        activation="sigmoid")(y)
        return Output

    def build(self):
        #Input = tf.keras.layers.Input(shape=(None, None, None, 1))
        Input = tf.keras.layers.Input(shape=(*config.default_shape,1))
        extra_filters = 0
        levels = list()
        levels.append(Input)
        levels[0] = Conv3D(filters=self.filter_count, kernel_size=3, strides=(1, 1, 1), padding="same")(levels[0])
        for i in range(self.level_count + 1):
            if self.residual:
                levels[i] = self.residual_section(levels[i], self.filter_count)
            else:
                levels[i] = self.conv_section(levels[i], self.filter_count)
            levels.append(MaxPooling3D(pool_size=(2, 2, 2), strides=None, padding='same')(levels[i]))
        x = levels[-1]
        for i in range(self.level_count, -1, -1):
            if self.concatenate:
                extra_filters = (self.level_count - i) * self.filter_count

            if self.residual:
                x = self.residual_section(x, self.filter_count + extra_filters)
            else:
                x = self.conv_section(x, self.filter_count + extra_filters)
            x = UpSampling3D((2, 2, 2))(x)
            if self.concatenate:
                x = concatenate([x, levels[i]], axis=-1)
            else:
                x = Add()([x, levels[i]])
        if self.concatenate:
            extra_filters = (self.level_count + 1) * self.filter_count
        if self.residual:
            Output = self.residual_end_section(x, self.filter_count + extra_filters)
        else:
            Output = self.conv_end_section(x, self.filter_count + extra_filters)
        self.model = Model(inputs=(Input), outputs=(Output))
        print(self.model.summary())
        self.model.compile(optimizer=self.optimizer,
                           loss=self.loss,
                           metrics=[metrics.mae, dice_coef])

    def save_model(self):
        local_path = ""
        model_json = self.model.to_json()
        with open(os.path.join(local_path, "model_unet.json"), "w") as json_file:
            json_file.write(model_json)

    def get_args(self):
        description = "{},{},{},{},{},{},{},{}".format(self.conv_count, self.filter_count, self.level_count,
                                                       self.dropout, self.optimizer, self.loss, self.output_dim,
                                                       self.residual)
        description += "\n lr=" + str(self.lr)
        return description


def conv_model(loss="mean_squared_error"):
    model = tf.keras.models.Sequential()
    model.add(Conv3D(filters=10, kernel_size=3, strides=(1, 1, 1), padding="same", activation="relu"))
    model.add(Conv3D(filters=10, kernel_size=3, strides=(1, 1, 1), padding="same", activation="relu"))
    model.add(Conv3D(filters=1, kernel_size=3, strides=(1, 1, 1), padding="same", activation="sigmoid"))
    # model.compile(loss=loss_fn_w_xe_logits(1), optimizer="sgd", metrics=["acc"])
    sgd = optimizers.SGD(lr=0.01, decay=1e-6, momentum=0.9, nesterov=True)
    model.compile(loss=loss, optimizer=sgd, metrics=["acc"])
    return model
    # model.compile(optimizer='adadelta', loss='categorical_crossentropy',
    #          metrics=['acc'])
    # return model


def conv_model_simple(loss="mean_squared_error"):
    model = tf.keras.models.Sequential()
    model.add(Conv3D(filters=1, kernel_size=3, strides=(1, 1, 1), padding="same", activation="sigmoid"))
    sgd = optimizers.SGD(lr=0.01, decay=1e-6, momentum=0.9, nesterov=True)
    model.compile(loss=loss, optimizer=sgd, metrics=["acc"])
    return model


def conv_model_deep():
    model = tf.keras.models.Sequential()

    model.add(Conv3D(filters=30, kernel_size=3, strides=(1, 1, 1), padding="same", activation="relu"))
    model.add(Conv3D(filters=30, kernel_size=3, strides=(1, 1, 1), padding="same", activation="relu"))
    model.add(Conv3D(filters=30, kernel_size=3, strides=(1, 1, 1), padding="same", activation="relu"))

    model.add(Conv3D(filters=30, kernel_size=3, strides=(1, 1, 1), padding="same", activation="relu"))
    model.add(Conv3D(filters=30, kernel_size=3, strides=(1, 1, 1), padding="same", activation="relu"))
    model.add(Conv3D(filters=30, kernel_size=3, strides=(1, 1, 1), padding="same", activation="relu"))

    model.add(Conv3D(filters=30, kernel_size=3, strides=(1, 1, 1), padding="same", activation="relu"))
    model.add(Conv3D(filters=30, kernel_size=3, strides=(1, 1, 1), padding="same", activation="relu"))
    model.add(Conv3D(filters=1, kernel_size=3, strides=(1, 1, 1), padding="same", activation="sigmoid"))

    model.compile(optimizer='adam',
                  loss='categorical_crossentropy',
                  metrics=['accuracy'])
    return model


def conv_model_auto(loss='mean_squared_error'):
    model = tf.keras.models.Sequential()

    model.add(Conv3D(filters=30, kernel_size=3, strides=(1, 1, 1), padding="same", activation="relu"))
    model.add(Conv3D(filters=30, kernel_size=3, strides=(1, 1, 1), padding="same", activation="relu"))
    model.add(Conv3D(filters=30, kernel_size=3, strides=(1, 1, 1), padding="same", activation="relu"))

    model.add(MaxPooling3D(pool_size=(2, 2, 2), strides=None, padding='valid'))

    model.add(Conv3D(filters=30, kernel_size=3, strides=(1, 1, 1), padding="same", activation="relu"))
    model.add(Conv3D(filters=30, kernel_size=3, strides=(1, 1, 1), padding="same", activation="relu"))
    model.add(Conv3D(filters=30, kernel_size=3, strides=(1, 1, 1), padding="same", activation="relu"))

    model.add(UpSampling3D((2, 2, 2)))

    model.add(Conv3D(filters=30, kernel_size=3, strides=(1, 1, 1), padding="same", activation="relu"))
    model.add(Conv3D(filters=30, kernel_size=3, strides=(1, 1, 1), padding="same", activation="relu"))
    model.add(Conv3D(filters=1, kernel_size=3, strides=(1, 1, 1), padding="same", activation="sigmoid"))

    model.compile(optimizer='sgd',
                  loss=loss,
                  metrics=['accuracy'])
    return model


def conv_model_auto_simple(loss='mean_squared_error'):
    model = tf.keras.models.Sequential()

    model.add(Conv3D(filters=30, kernel_size=3, strides=(1, 1, 1), padding="same", activation="relu"))

    model.add(MaxPooling3D(pool_size=(2, 2, 2), strides=None, padding='valid'))

    model.add(Conv3D(filters=30, kernel_size=3, strides=(1, 1, 1), padding="same", activation="relu"))

    model.add(UpSampling3D((2, 2, 2)))

    model.add(Conv3D(filters=30, kernel_size=3, strides=(1, 1, 1), padding="same", activation="relu"))
    model.add(Conv3D(filters=1, kernel_size=3, strides=(1, 1, 1), padding="same", activation="sigmoid"))

    sgd = optimizers.SGD(lr=0.01, decay=1e-6, momentum=0.9, nesterov=True)
    model.compile(loss=loss, optimizer=sgd, metrics=["accuracy"])
    # model.compile(optimizer='sgd',
    #           loss='categorical_crossentropy',
    #          metrics=['accuracy'])
    return model


def conv_no_pooling(optimizer="sgd", loss='mean_squared_error'):
    model = tf.keras.models.Sequential()

    model.add(Conv3D(filters=30, kernel_size=5, strides=(2, 2, 2), padding="valid", activation="relu"))
    model.add(Conv3D(filters=30, kernel_size=5, strides=(2, 2, 2), padding="valid", activation="relu"))
    model.add(Conv3D(filters=30, kernel_size=5, strides=(2, 2, 2), padding="valid", activation="relu"))

    # model.add(Conv3DTranspose

    model.compile(optimizer='sgd',
                  loss=loss,
                  metrics=[metrics.mae])
    return model


def conv_model_auto_deep(optimizer="sgd", loss='mean_squared_error', output_dim=1):
    model = tf.keras.models.Sequential()

    model.add(Conv3D(filters=30, kernel_size=3, strides=(1, 1, 1), padding="same", activation="relu"))
    model.add(Conv3D(filters=30, kernel_size=3, strides=(1, 1, 1), padding="same", activation="relu"))
    model.add(Conv3D(filters=30, kernel_size=3, strides=(1, 1, 1), padding="same", activation="relu"))

    model.add(MaxPooling3D(pool_size=(2, 2, 2), strides=None, padding='valid'))

    model.add(Conv3D(filters=30, kernel_size=3, strides=(1, 1, 1), padding="same", activation="relu"))
    model.add(Conv3D(filters=30, kernel_size=3, strides=(1, 1, 1), padding="same", activation="relu"))
    model.add(Conv3D(filters=30, kernel_size=3, strides=(1, 1, 1), padding="same", activation="relu"))

    model.add(MaxPooling3D(pool_size=(2, 2, 2), strides=None, padding='valid'))

    model.add(Conv3D(filters=30, kernel_size=3, strides=(1, 1, 1), padding="same", activation="relu"))
    model.add(Conv3D(filters=30, kernel_size=3, strides=(1, 1, 1), padding="same", activation="relu"))
    model.add(Conv3D(filters=30, kernel_size=3, strides=(1, 1, 1), padding="same", activation="relu"))

    model.add(UpSampling3D((2, 2, 2)))

    model.add(Conv3D(filters=30, kernel_size=3, strides=(1, 1, 1), padding="same", activation="relu"))
    model.add(Conv3D(filters=30, kernel_size=3, strides=(1, 1, 1), padding="same", activation="relu"))
    model.add(Conv3D(filters=30, kernel_size=3, strides=(1, 1, 1), padding="same", activation="relu"))

    model.add(UpSampling3D((2, 2, 2)))

    model.add(Conv3D(filters=30, kernel_size=3, strides=(1, 1, 1), padding="same", activation="relu"))
    model.add(Conv3D(filters=30, kernel_size=1, strides=(1, 1, 1), padding="same", activation="relu"))
    model.add(Conv3D(filters=output_dim, kernel_size=1, strides=(1, 1, 1), padding="same", activation="sigmoid"))

    model.compile(optimizer=optimizer,
                  loss=loss,
                  metrics=[metrics.mae])
    return model


def unet(optimizer="sgd", loss='mean_squared_error', output_dim=1):
    Input = tf.keras.layers.Input(shape=(None, None, None, 1))  # For debugging: shape=(40,40,40,1)

    x = Conv3D(filters=30, kernel_size=3, strides=(1, 1, 1), padding="same", activation="relu")(Input)
    x = Conv3D(filters=30, kernel_size=3, strides=(1, 1, 1), padding="same", activation="relu")(x)
    x = Conv3D(filters=30, kernel_size=3, strides=(1, 1, 1), padding="same", activation="relu")(x)

    y = MaxPooling3D(pool_size=(2, 2, 2), strides=None, padding='valid')(x)

    y = Conv3D(filters=30, kernel_size=3, strides=(1, 1, 1), padding="same", activation="relu")(y)
    y = Conv3D(filters=30, kernel_size=3, strides=(1, 1, 1), padding="same", activation="relu")(y)
    y = Conv3D(filters=30, kernel_size=3, strides=(1, 1, 1), padding="same", activation="relu")(y)

    y = MaxPooling3D(pool_size=(2, 2, 2), strides=None, padding='valid')(y)

    y = Conv3D(filters=30, kernel_size=3, strides=(1, 1, 1), padding="same", activation="relu")(y)
    y = Conv3D(filters=30, kernel_size=3, strides=(1, 1, 1), padding="same", activation="relu")(y)
    y = Conv3D(filters=30, kernel_size=3, strides=(1, 1, 1), padding="same", activation="relu")(y)

    y = UpSampling3D((2, 2, 2))(y)

    y = Conv3D(filters=30, kernel_size=3, strides=(1, 1, 1), padding="same", activation="relu")(y)
    y = Conv3D(filters=30, kernel_size=3, strides=(1, 1, 1), padding="same", activation="relu")(y)
    y = Conv3D(filters=30, kernel_size=3, strides=(1, 1, 1), padding="same", activation="relu")(y)

    y = UpSampling3D((2, 2, 2))(y)

    y = concatenate([x, y], axis=-1)

    y = Conv3D(filters=60, kernel_size=3, strides=(1, 1, 1), padding="same", activation="relu")(y)
    y = Conv3D(filters=40, kernel_size=1, strides=(1, 1, 1), padding="same", activation="relu")(y)
    Output = Conv3D(filters=output_dim, kernel_size=1, strides=(1, 1, 1), padding="same", activation="sigmoid")(y)

    model = Model(inputs=(Input), outputs=(Output))
    # for layer in model.layers:
    #    print(layer.output_shape)
    model.compile(optimizer=optimizer,
                  loss=loss,
                  metrics=[metrics.mae])
    return model


def model_sol():
    pass


models = [conv_model, conv_model_deep, conv_model_auto]

if __name__=="__main__":
    unet = UNet(concatenate=True, filter_count=30)
    unet.build()