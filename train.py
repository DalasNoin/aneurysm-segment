from config import PATH
import preprocessing
import tensorflow as tf
import keras
import net
from tensorflow.keras.layers import Conv3D, Conv3DTranspose, Flatten, Dense

ad = preprocessing.AneurysmData(PATH,size = (40,40,40))

for model_function in net.models:
    model = model_function()
    x = ad.correct_shape(ad.images)
    y = ad.correct_shape(ad.masks)
    model.fit(x,y,batch_size=10,epochs=10, validation_split = 0.1)