import numpy as np
from tensorflow.keras import backend as K



def dice_coef(y_true, y_pred):
    eps = 0.1
    y_true_f = K.flatten(y_true)
    y_pred_f = K.flatten(y_pred)
    intersection = K.sum(y_true_f * y_pred_f)
    return (2. * intersection + eps) / (K.sum(y_true_f**2) + K.sum(y_pred_f**2) + eps)

def dice(y_true, y_pred):
    y_true = np.ndarray.flatten(y_true)
    y_pred = np.ndarray.flatten(y_pred)
    y_pred = np.where(y_pred>0.5,1,0)
    
    intersection = np.dot(y_true,y_pred)
    return (2*intersection)/(np.sum(y_true)+np.sum(y_pred))


def dice_coef_loss(y_true, y_pred):
    return 1.0 - dice_coef(y_true, y_pred)