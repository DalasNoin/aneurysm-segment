from tensorflow import keras
from tensorflow.keras import losses
from tensorflow.keras.backend import K

import numpy as np
from data import patching
import config


def dice_coef(y_true, y_pred):
    y_true_f = K.flatten(y_true)
    y_pred_f = K.flatten(y_pred)
    intersection = K.sum(y_true_f * y_pred_f)
    return (2. * intersection + 1e-7) / (K.sum(y_true_f) + K.sum(y_pred_f) + 1e-7)


class Prediction:
    def __init__(self):
        self.model = keras.models.load_model(config.MODEL_PATH,
                                             custom_objects={"loss": losses.mean_squared_error, "dice_coef": dice_coef})

    def forward(self, tensor):
        shape = tensor.shape
        output = np.zeros(shape)
        pi = patching.PatchImage(name="input", tensor=tensor, shape=config.default_shape, stride=config.default_shape)

        for patch_tensor, indices, patch in pi.iterate():
            output[patch[0]:patch[3], patch[1]:patch[4], patch[2]:patch[5]] = self.model.predict(patch_tensor)

        return output
