import os
import numpy as np
from scipy import ndimage
from matplotlib import pyplot as plt

class PatchImage:
    def __init__(self,name, tensor, shape=(40,40,40),stride=(35,35,35)):
        self.patch_indices = self.patch_3d(tensor,shape,stride)
        self.name = name

    def patch_3d(self, tensor, shape, strides):
        windows = [int(np.ceil((x - (x_patch // 2)) / x_stride)) for x, x_patch, x_stride in
                   zip(tensor.shape, shape, strides)]
        return windows

    def get_patch(self, indices, tensor, shape, strides):
        xyz_begin = [min(x_shape, x * stride) for x, stride, x_shape in zip(indices, strides, tensor.shape)]
        xyz_end = [min(x_shape, x + x_patch) for x, x_patch, x_shape in zip(xyz_begin, shape, tensor.shape)]
        xyz = xyz_begin + xyz_end
        return xyz

    def cut_patch(self, tensor, patch):
        return tensor[patch[0]:patch[3], patch[1]:patch[4], patch[2]:patch[5]]




class Patch:
    def __init__(self, parent_name, patch, patch_pixels, patch_indices):
        self.parent_name = parent_name
        self.patch = patch
        self.patch_pixels = patch_pixels
        self.patch_indices = patch_indices

    def info(self):
        return "{},{},{},{},{},"
