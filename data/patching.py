import os
import numpy as np
from scipy import ndimage
from matplotlib import pyplot as plt


class PatchImage:
    def __init__(self, name, tensor, shape=(40, 40, 40), stride=(35, 35, 35)):
        self.tensor = tensor
        self.shape = shape
        self.stride = stride
        self.name = name
        self.patch_indices = self.patch_3d()

    def patch_3d(self):
        windows = [int(np.ceil((x - (x_patch // 2)) / x_stride)) for x, x_patch, x_stride in
                   zip(self.tensor.shape, self.shape, self.stride)]
        return windows

    def get_patch(self, indices):
        xyz_begin = [min(x_shape, x * stride) for x, stride, x_shape in zip(indices, self.stride, self.tensor.shape)]
        xyz_end = [min(x_shape, x + x_patch) for x, x_patch, x_shape in zip(xyz_begin, self.shape, self.tensor.shape)]
        xyz = xyz_begin + xyz_end
        return xyz

    def cut_patch(self, patch):
        return self.tensor[patch[0]:patch[3], patch[1]:patch[4], patch[2]:patch[5]]



    def iterate(self):
        X, Y, Z = np.meshgrid(*[np.arange(value) for value in self.patch_indices])
        X, Y, Z = X.flatten(), Y.flatten(), Z.flatten()
        for indices in zip(X, Y, Z):
            patch = self.get_patch(indices)
            yield self.cut_patch(patch), indices, patch


class Patch:
    def __init__(self, parent_name, patch, patch_pixels, patch_indices):
        self.parent_name = parent_name
        self.patch = patch
        self.patch_pixels = patch_pixels
        self.patch_indices = patch_indices

    def info(self):
        return "{},{},{},{},{},"
