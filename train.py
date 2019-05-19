from config import PATH
import preprocessing
import tensorflow as tf
import keras
import net
from tensorflow.keras.layers import Conv3D, Conv3DTranspose, Flatten, Dense

ad = preprocessing.AneurysmData(PATH,size = (20,20,20))
model = net.conv_auto_model()
x = ad.correct_shape(ad.images)
y = ad.correct_shape(ad.masks)
model.fit(x,y,batch_size=10,epochs=10)