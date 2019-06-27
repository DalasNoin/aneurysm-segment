from tensorflow import keras
from tensorflow.keras import losses

import numpy as np
from data import patching, sequence_generators
import config
from tensorflow.keras.models import load_model

import net



class Prediction:
    def __init__(self):
        self.model = load_model(config.MODEL_PATH,
                                             custom_objects={"loss": net.weighted_crossentropy(500), "dice_coef": net.dice_coef})

    def forward(self, tensor):
        shape = tensor.shape
        output = np.zeros(shape)
        pi = patching.PatchImage(name="input", tensor=tensor, shape=config.default_shape, stride=config.default_shape)

        for patch_tensor, indices, patch in pi.iterate():
            output[patch[0]:patch[3], patch[1]:patch[4], patch[2]:patch[5]] = self.model.predict(patch_tensor)

        return output

def evaluate_model(model_path):
    model = load_model(model_path,custom_objects={"loss":net.weighted_crossentropy(500),"dice_coef":net.dice_coef})

    partition = sequence_generators.get_train_val_sequence()
    test_gen = sequence_generators.DataGenerator(partition["test"],config.patch_validation_data_path, batch_size=2)
    print(model.evaluate_generator(test_gen))
    print(model.metrics_names)
    
if __name__=="__main__":
    evaluate_model(config.model_path)
    