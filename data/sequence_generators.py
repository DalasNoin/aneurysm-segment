from tensorflow.python.keras.utils.data_utils import Sequence
import config
import os
import pandas as pd
import numpy as np
from tensorflow import keras


def get_train_val_sequence():
    records_train = pd.read_csv(os.path.join(config.patch_data_path, "records.csv"))
    records_validation = pd.read_csv(os.path.join(config.patch_validation_data_path, "records.csv"))
    names = list(records_train[records_train["positiv"]]["filepath"])
    names_train = [name.split("/")[-1] + ".npy" for name in names]
    names_validation = [name.split("/")[-1] + ".npy" for name in records_validation["filepath"]]
    return {"test":names_validation, "train":names_train}


class DataGenerator(Sequence):
    def __init__(self, names, base_path, batch_size=20, shuffle=True, flip=True):
        self.names = names
        self.base_path = base_path
        self.shuffle = shuffle
        self.batch_size = batch_size
        self.on_epoch_end()
        self.flip=flip
        if flip:
            self.flip_choices = [None,(0),(1),(2),(0, 1),(0, 2),(1, 2),(0,1,2)]

    def __len__(self):
        return int(np.floor(len(self.names) / self.batch_size))

    def __getitem__(self, index):
        indexes = self.indexes[index * self.batch_size:(index + 1) * self.batch_size]

        # Find list of IDs
        names_temp = [self.names[k] for k in indexes]

        # Generate data
        X, y = self.__data_generation(names_temp)

        return X, y
    
    def get_sample(self):
        name = self.names[-1]
        return self.__data_generation([name])[0][:1],self.__data_generation([name])[1][:1]

    def on_epoch_end(self):
        'Updates indexes after each epoch'
        self.indexes = np.arange(len(self.names))
        if self.shuffle:
            np.random.shuffle(self.indexes)

    def __data_generation(self, names_temp):
        X = np.zeros((self.batch_size, *config.shape, 1))
        y = np.zeros((self.batch_size, *config.shape, 1), dtype=int)

        # Generate data
        for i, ID in enumerate(names_temp):
            # Store sample
            temp = np.load(self.base_path+'/image/' + ID)
            if self.flip:
                flip_axis=np.random.choice(self.flip_choices)
            
            if self.flip and flip_axis is not None:
                np.flip(temp,axis=flip_axis)
            X[i, :temp.shape[0], :temp.shape[1], :temp.shape[2],:] = temp.reshape(*temp.shape, 1)

            # Store class
            temp = np.load(self.base_path+'/mask/' + ID)
            if self.flip and flip_axis is not None:
                np.flip(temp,axis=flip_axis)
            y[i, :temp.shape[0], :temp.shape[1], :temp.shape[2],:] = temp.reshape(*temp.shape,1)
        
        return X, y
