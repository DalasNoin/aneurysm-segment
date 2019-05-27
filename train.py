import tensorflow as tf
import keras
from tensorflow.keras.layers import Conv3D, Conv3DTranspose, Flatten, Dense
import numpy as np
%load_ext autoreload
%autoreload 2
%aimport data
%reload_ext data
import data
from config import patch_data_path
import preprocessing
import os
import net
import pandas as pd
records = pd.read_csv(os.path.join(patch_data_path,"records.csv"))
names = list(records[records["positiv"]]["filepath"])
names = [name.split("/")[-1]+".npy" for name in names]

adg = preprocessing.AneurysmDataGenerator(patch_data_path,names)

model = net.conv_model_auto_deep()

x = adg.images#[image for i, image in enumerate(adg.images) if np.mean(adg.masks[i])>0.01]
x = np.array(x)
x = x.reshape(*x.shape,1)
y = adg.images#[mask for i, image in enumerate(adg.masks) if np.mean(adg.masks[i])>0.01]
y = np.array(y)
y = y.reshape(*y.shape,1)

model.fit(x,y,batch_size=20,epochs=10,validation_split=0.05)
