import pydicom
import os
import numpy as np
from os import listdir
from os.path import isfile, join
from config import path_images, path_masks

path_combined = "/home/bagel/data/aneurysms/imageData/combined/image/"

def discover(path):
    return [join(path, f) for f in listdir(path) if isfile(join(path, f))]

def extract_pixel_arrays(path_list):
    ret = list()
    for path in path_list:
        ret.append(_extract_pixel_array(path))
    return ret

def _extract_pixel_array(path):
    image = pydicom.read_file(path)
    return image.pixel_array

def preprocess_x_data(X):
    X = [arr.reshape(*arr.shape,1) for arr in X]
    