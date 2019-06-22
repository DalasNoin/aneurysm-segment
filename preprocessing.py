import os
import numpy as np
from scipy import ndimage
from matplotlib import pyplot as plt
import pydicom
from os import listdir
from os.path import isfile, join
import config
from data import patching
import pandas as pd
from tqdm import tqdm


def discover(path):
    return [join(path, f) for f in listdir(path) if isfile(join(path, f))], [f for f in listdir(path) if
                                                                             isfile(join(path, f))]


def extract_pixel_arrays(path_list):
    ret = list()
    for path in path_list:
        ret.append(extract_pixel_array(path))
    return ret


def extract_pixel_array(path):
    if path[-3:] == "npy":
        return np.load(path)
    image = pydicom.read_file(path)
    return image.pixel_array


def preprocess_x_data(X):
    X = [arr.reshape(*arr.shape, 1) for arr in X]


def safe_mkdir(path):
    if not os.path.exists(path):
        os.mkdir(path)


class AneurysmData:
    def __init__(self, path, size=None, validation=False, mock=False):
        if not mock:
            path_image = os.path.join(path, "image")
            path_mask = os.path.join(path, "mask")

            self.paths_image, self.image_names = discover(path_image)
            self.paths_mask, self.mask_names = discover(path_mask)
            if not validation:
                self.paths_image = self.paths_image[config.validation_split:]
                self.paths_mask = self.paths_mask[config.validation_split:]
                self.image_names = self.image_names[config.validation_split:]
                self.mask_names = self.mask_names[config.validation_split:]
            else:
                self.paths_image = self.paths_image[:config.validation_split]
                self.paths_mask = self.paths_mask[:config.validation_split]
                self.image_names = self.image_names[:config.validation_split]
                self.mask_names = self.mask_names[:config.validation_split]
                
            if size is not None:
                self.images = self.downsampling(self.paths_image, size=size)
                self.masks = self.downsampling(self.paths_mask, size=size)
                # self.images = self.normalize(self.images)
                # self.masks = self.to_binary(self.masks).astype("int")
            # else:
            # self.images = self.load(self.paths_image)
            # self.masks = self.load(self.paths_mask)
            # self.images = self.normalize(self.images)
            # self.masks = self.to_binary(self.masks)

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

    def load_data(self, count=100):
        self.images = self.load(self.paths_image, count=count)
        self.masks = self.load(self.paths_mask, count=count)

    def load(self, paths, count=100):
        images = list()
        for path in paths[:count]:
            image = extract_pixel_array(path)
            images.append(image)
        return images

    def element_wise_save(self):
        safe_mkdir(config.full_data_path)
        mask_dir = join(config.full_data_path, "mask")
        image_dir = join(config.full_data_path, "image")
        safe_mkdir(mask_dir)
        safe_mkdir(image_dir)
        for image_path, image_name, mask_path, mask_name in zip(self.paths_image, self.image_names, self.paths_mask,
                                                                self.mask_names):
            image = extract_pixel_array(image_path)
            image = self.normalize(image).astype("float32")
            self.save_single(image_dir, image_name, image)
            mask = extract_pixel_array(mask_path)
            mask = self.to_binary(mask)
            self.save_single(mask_dir, mask_name, mask)

    def save_patches_ew(self, threshold = 0.02, shape=(40,40,40), stride=(40,40,40), target_path=config.patch_data_path):
        safe_mkdir(target_path)
        self.threshold = threshold
        safe_mkdir(config.full_data_path)
        mask_dir = join(target_path, "mask")
        image_dir = join(target_path, "image")
        safe_mkdir(mask_dir)
        safe_mkdir(image_dir)
        print("Create Patches for the images")
        self.records = pd.DataFrame(self.run_patch_pos(target_path, shape=shape, stride=stride),columns=["Indices", "patches", "filepath", "name", "positiv"])
        #self.run_patch(self.paths_image, self.image_names, image_dir)
        print("Create patches for the masks")
        #self.records = pd.DataFrame(self.run_patch(self.paths_mask, self.mask_names, mask_dir, mask=True),
                                    #columns=["Indices", "patches", "filepath", "name", "positiv"])
        self.records.to_csv(join(target_path, "records.csv"))

    def generate_filepath(self, target_dir, name, indices):
        if "." in name:
            name = name.split(".")[0]
        filepath = join(target_dir, "{}-{}x{}x{}".format(name, *indices))
        return filepath
    
    
    def run_patch_pos(self, target_dir, shape=(40,40,40),stride=(40,40,40)):
        records = []
        mask_dir = join(target_dir, "mask")
        image_dir = join(target_dir, "image")
        for image_path, image_name, mask_path, mask_name in tqdm(zip(self.paths_image, self.image_names, self.paths_mask, self.mask_names), total=len(self.paths_image)):
            image = extract_pixel_array(image_path)
            mask = extract_pixel_array(mask_path)
            pi = patching.PatchImage(name=image_name, tensor=image,stride=stride, shape=shape)
            pi_mask = patching.PatchImage(name=mask_name, tensor=mask,stride=stride, shape=shape)
            for mask_sub_tensor, indices, patch in pi_mask.iterate():
                image_filepath = self.generate_filepath(image_dir, image_name, indices)
                mask_filepath = self.generate_filepath(mask_dir, image_name, indices)
                
                positive_eg = np.mean(mask_sub_tensor) > self.threshold
                if not positive_eg: continue
                sub_tensor = pi.cut_patch(patch)
                records.append([indices, patch, mask_filepath, image_name, positive_eg])
                np.save(image_filepath, self.normalize(sub_tensor))
                np.save(mask_filepath, mask_sub_tensor)
        return records

    def run_patch(self, paths, names, target_dir, mask=False):
        if mask:
            records = []
        for path, name in tqdm(zip(paths, names), total=len(paths)):
            image = extract_pixel_array(path)
            pi = patching.PatchImage(name=name, tensor=image)
            for sub_tensor, indices, patch in pi.iterate():
                filepath = self.generate_filepath(target_dir, name, indices)
                if mask:
                    positive_eg = len(np.unique(sub_tensor)) != 1
                    records.append([indices, patch, filepath, name, positive_eg])
                np.save(filepath, sub_tensor)
        if mask:
            return records

    def downsampling(self, paths, size=(16, 16, 16)):
        images = list()
        for path in paths:
            image = extract_pixel_array(path)
            images.append(self.downsample_image(image, size))
        return np.array(images)

    def normalize(self, data):
        """
        normalizes based on entire data, may change to by data point
        :param data:
        :return:
        """
        if isinstance(data, list):
            return [(element - np.mean(element)) / np.std(element) for element in data]
        return (data - np.mean(data)) / np.std(data)

    def to_binary(self, data):
        if isinstance(data, list):
            return [(element != 0).astype("int") for element in data]
        return data != 0

    def correct_shape(self, data):
        return data.reshape(*data.shape, 1)

    def show_image(self, image_idx=0, N=4):
        image = self.images[image_idx]
        mask = self.masks[image_idx]
        plt.figure(dpi=800)
        plt.rcParams['figure.figsize'] = [20, 20]
        fig, ax = plt.subplots(nrows=N, ncols=N)
        for i in range(N):
            for j in range(N):
                im = image[i + j * N, :, :]
                m = mask[i + j * N, :, :]
                ax[i, j].imshow(im, alpha=0.2, cmap="Greys_r")
                ax[i, j].imshow(m == 0, alpha=0.8, cmap="cividis", vmin=0, vmax=1)
        return fig

    def save_single(self, path, name, tensor):
        new_path = join(path, name)
        np.save(new_path, tensor)

    def save(self, path, names, tensors):
        new_paths = [join(path, name) for name in names]
        for tensor, path in zip(tensors, new_paths):
            np.save(path, tensor)

    def save_all(self, path):
        safe_mkdir(path)
        self.save(path, self.image_names, self.images)
        self.save(path, self.mask_names, self.masks)


class AneurysmDataGenerator:
    def __init__(self, base_path, names):
        self.base_path = base_path
        self.names = names
        self.get_paths()
        self.images = self.load(self.image_paths)
        self.masks = self.load(self.mask_paths)

    def get_paths(self):
        self.mask_paths = [join(self.base_path, "mask", name) for name in self.names]
        self.image_paths = [join(self.base_path, "image", name) for name in self.names]

    def load(self, paths):
        images = list()
        for path in tqdm(paths):
            image = extract_pixel_array(path)
            images.append(image)
        return images

    def show_image(self, image_idx=0, N=4, alpha_image=0.5, alpha_mask=0.5):
        image = self.images[image_idx]
        mask = self.masks[image_idx]
        plt.figure(dpi=800)
        plt.rcParams['figure.figsize'] = [20, 20]
        fig, ax = plt.subplots(nrows=N, ncols=N)
        for i in range(N):
            for j in range(N):
                im = image[i + j * N, :, :]
                m = mask[i + j * N, :, :]
                ax[i, j].imshow(im, alpha=alpha_image, cmap="Greys_r")
                ax[i, j].imshow(m == 0, alpha=alpha_mask, cmap="cividis", vmin=0, vmax=1)
        return fig

def get_aneurysm_generator():
    records = pd.read_csv(os.path.join(config.patch_data_path,"records.csv"))
    names = list(records[records["positiv"]]["filepath"])
    names = [name.split("/")[-1]+".npy" for name in names]
    adg = AneurysmDataGenerator(config.patch_data_path,names)
    return adg

def make_validation_data():
    ad = AneurysmData(config.full_data_path,validation=True)
    ad.save_patches_ew(shape=config.shape,threshold=-0.0001, stride=config.stride, target_path=config.patch_validation_data_path)
    
def make_train_data():
    ad = AneurysmData(config.full_data_path,validation=False)
    ad.save_patches_ew(shape=config.shape,threshold=0.0001, stride=config.stride, target_path=config.patch_data_path)

if __name__ == "__main__":
    # ad = AneurysmData(config.PATH, size=None)
    # ad.element_wise_save()
    # ad = AneurysmData(config.full_data_path)
    # ad.save_patches_ew()
    # ad.records.to_csv(join(config.patch_data_path, "records.csv"))
    make_validation_data()
    make_train_data()
    #ad = AneurysmData(config.full_data_path)
    #ad.save_patches_ew(shape=config.shape,threshold=0.0001, stride=config.stride)
