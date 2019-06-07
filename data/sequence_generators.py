from keras.utils import Sequence
import config
import os
import pandas as pd
import numpy as np
from tensorflow import keras


def get_train_val_sequence(ratio=0.05):
    records = pd.read_csv(os.path.join(config.patch_data_path, "records.csv"))
    names = list(records[records["positiv"]]["filepath"])
    names = [name.split("/")[-1] + ".npy" for name in names]
    val_split = int(len(names) * ratio)
    return names[:val_split], names[val_split:]


class DataGenerator(Sequence):
    def __init__(self, names, batch_size=20, shuffle=True):
        self.names = names
        self.shuffle = shuffle
        self.batch_size = batch_size
        self.on_epoch_end()

    def __len__(self):
        return int(np.floor(len(self.names) / self.batch_size))

    def __getitem__(self, index):
        indexes = self.indexes[index * self.batch_size:(index + 1) * self.batch_size]

        # Find list of IDs
        names_temp = [self.names[k] for k in indexes]

        # Generate data
        X, y = self.__data_generation(names_temp)

        return X, y

    def on_epoch_end(self):
        'Updates indexes after each epoch'
        self.indexes = np.arange(len(self.names))
        if self.shuffle:
            np.random.shuffle(self.indexes)

    def __data_generation(self, names_temp):
        X = np.empty((self.batch_size, None, None, None, 1))
        y = np.empty((self.batch_size, None, None, None, 1), dtype=int)

        # Generate data
        for i, ID in enumerate(names_temp):
            # Store sample
            temp = np.load(config.patch_data_path+'/image/' + ID)
            X[i, ] = temp.reshape(*temp.shape, 1)

            # Store class
            temp = np.load(config.patch_data_path+'/mask/' + ID)
            y[i, ] = temp.reshape(*temp.shape,1)

        return X, y
