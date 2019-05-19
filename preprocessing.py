import data
import os
import numpy as np
from scipy import ndimage
from matplotlib import pyplot as plt


class AneurysmData:
    def __init__(self, path, size=(16,16,16), mock=False):
        if not mock:
            path_image = os.path.join(path, "image")
            path_mask = os.path.join(path, "mask")
            self.paths_image = data.discover(path_image)
            self.paths_mask = data.discover(path_mask)
            self.images = self.downsampling(self.paths_image, size = size)
            self.images = self.renormalize(self.images)
            self.masks = self.downsampling(self.paths_mask, size = size)
            self.masks = self.to_binary(self.masks).astype("int")
        else:
            self.images = [np.random.randint(low=-100, high=100, size=(1, 110, 256 + i, 256 - i)) for i in range(100)]
            self.masks = [np.random.randint(low=0, high=1, size=(1, 110, 256 + i, 256 - i)) for i in
                          range(100)]
            # self.images = np.random.random_integers(low=-100, high=100, size=(100, 220, 256, 256))
            # self.masks = np.random.random_integers(low=0, high=1, size=(100, 220, 256, 256))

    def downsample_image(self, image, size=(16, 16, 16)):
        zoom = [target / source for target, source in zip(size, image.shape)]
        image = ndimage.zoom(image, zoom=zoom)
        return image

    def downsampling(self, paths, size=(16, 16, 16)):
        images = list()
        for path in paths:
            image = data.extract_pixel_array(path)
            images.append(self.downsample_image(image, size))
        return np.array(images)
    
    def renormalize(self, data):
        return (data - np.mean(data))/np.std(data)
    
    def to_binary(self, data):
        return data != 0
    
    def correct_shape(self,data):
        return data.reshape(*data.shape,1)
    
    def show_image(self,image_idx=0, N=4):
        image = self.images[image_idx]
        mask = self.masks[image_idx]
        plt.figure(dpi=800)
        plt.rcParams['figure.figsize'] = [20,20]
        fig, ax = plt.subplots(nrows=N, ncols=N)
        for i in range(N):
            for j in range(N):
                im = image[i+j*N,:,:]
                m = mask[i+j*N,:,:]
                ax[i,j].imshow(im,alpha=0.2,cmap="Greys_r")
                ax[i,j].imshow(m==0,alpha=0.8, cmap="cividis",vmin=0,vmax=1)
        return fig