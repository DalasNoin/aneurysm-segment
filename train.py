

import tensorflow as tf
from tensorflow.keras.layers import Conv3D, Conv3DTranspose, Flatten, Dense
import numpy as np
import data
from config import patch_data_path
import config
import preprocessing
import os
import net
import pandas as pd
from mpl_toolkits.mplot3d import Axes3D
import matplotlib.pyplot as plt
import matplotlib.animation as animation
import datetime
from tensorflow.keras.utils import plot_model
from data import sequence_generators
import argparse
from tensorflow.keras.callbacks import TensorBoard, ModelCheckpoint
from time import time
from custom_layers import norm

from tensorflow.keras.models import load_model

class Trainer:
    def __init__(self):
        os.makedirs("plots",exist_ok=True)
        os.makedirs("logs",exist_ok=True)
    
    def train(self, batch_size, epochs=10, optimizer="adam", loss = "mean_squared_error", name="TR", model=None, descriptive_args=""):
        self.loss=loss
        self.batch_size = batch_size
        self.epochs = epochs
        self.optimizer = optimizer
        self.name = "{}{}".format(datetime.datetime.now().strftime("%Y%m%dx%H%M%S"),name)
        self.descriptive_args = descriptive_args
        self.log_path = "logs/{}/".format(self.name)
        os.makedirs(self.log_path,exist_ok=True)
        #records = pd.read_csv(os.path.join(patch_data_path,"records.csv"))
        #names = list(records[records["positiv"]]["filepath"])
        #names = [name.split("/")[-1]+".npy" for name in names]

        #self.adg = preprocessing.AneurysmDataGenerator(patch_data_path,names)

        

        #self.x = self.adg.images
        #self.x = np.array(self.x)
        #self.x = self.x.reshape(*self.x.shape,1)
        #self.y = self.adg.masks
        #self.y = np.array(self.y)
        
        
        #if "crossentropy" in self.loss:
            
            #self.y = keras.utils.to_categorical(self.y)
            
            #self.model = net.conv_model_auto_deep(loss=loss, optimizer=optimizer, output_dim=2)
        #else:
            
        #self.y = self.y.reshape(*self.y.shape,1)
            #self.model = net.conv_model_auto_deep(loss=loss, optimizer=optimizer)
        if model is None:
            self.model = net.unet(loss=loss, optimizer=optimizer)
        else:
            self.model=model
        partition = sequence_generators.get_train_val_sequence()
        self.train_gen = sequence_generators.DataGenerator(partition["train"],config.patch_data_path, batch_size=self.batch_size, flip=True)
        self.test_gen = sequence_generators.DataGenerator(partition["test"],config.patch_validation_data_path, batch_size=self.batch_size, flip=False)
        os.makedirs(self.log_path+"tensorboard_logs/",exist_ok=True)
        tensorboard = TensorBoard(log_dir=self.log_path+"tensorboard_logs/{}".format(time()))
        os.makedirs(self.log_path+"chkpnts/",exist_ok=True)
        modelcheckpoint = ModelCheckpoint(self.log_path+"chkpnts/{epoch:03d}-vd{val_dice_coef:.5f}.hdf5", monitor='val_dice_coef', verbose=0, save_best_only=False, save_weights_only=False, mode='auto', period=1)
        self.history = self.model.fit_generator(generator=self.train_gen,epochs=epochs,validation_data=self.test_gen,callbacks=[modelcheckpoint, tensorboard])
        self.log()
        
    def plot_history(self,path):
        for col in ['dice_coef', 'val_dice_coef']:
            plt.plot(self.history.history[col],label=col)
        plt.legend()
        plt.ylim((-0.05,1.05))
        plt.savefig(os.path.join(path,"stats.pdf"))
        
    def log(self):
        local_path = "logs/{}".format(self.name)
        pd.DataFrame(self.history.history).to_csv(os.path.join(local_path, "history.csv"))
        args = "{},{},{},{}".format(self.name,self.epochs, self.optimizer,self.loss)
        args += "\nDescription: {}".format(self.descriptive_args)
        with open(os.path.join(local_path,"args.txt"),"w") as f:
            f.write(args)
            
        self.plot_history(local_path)
            
        model_json = self.model.to_json()
        with open(os.path.join(local_path,"model.json"), "w") as json_file:
            json_file.write(model_json)
        self.model.save(os.path.join(local_path,'model.h5'))
        #plot_model(self.model, to_file=os.path.join(local_path,"model.png"))
        

    def sample_prediction(self, num=-1):
        #result = self.model.predict(self.x[num:num+1])
        #if False and "crossentropy" in self.loss or True:
        #    return result[...,1].reshape((40,40,40))
        result, mask = self.test_gen.get_sample()
        result = self.model.predict(result)
        print(result.shape, mask.shape)
        return result.reshape(config.shape) > 0.4, mask.reshape(config.shape)
    
    def plot_result(self):
        result = self.sample_prediction()
        N=4
        image = self.adg.masks[-1]
        mask = result >0.98
        plt.figure(dpi=800)
        plt.rcParams['figure.figsize'] = [20,20]
        fig, ax = plt.subplots(nrows=N, ncols=N)
        for i in range(N):
            for j in range(N):
                im = image[i+j*N+16,:,:]
                m = mask[i+j*N+16,:,:]
                ax[i,j].imshow(im,alpha=0.4,cmap="Greys_r",vmin=0,vmax=1)
                ax[i,j].imshow(m,alpha=0.6, cmap="cividis",vmin=0,vmax=1)
        plt.savefig("plots/plot.png")
         
            
    def midpoints(self, x):
        sl = ()
        for i in range(x.ndim):
            x = (x[sl + np.index_exp[:-1]] + x[sl + np.index_exp[1:]]) / 2.0
            sl += np.index_exp[:]
        return x
    
    def plot_result3D(self):
        result_mask = self.sample_prediction() > 0.95
        mask = self.adg.masks[-1]
        # prepare some coordinates
        x, y, z = np.indices((41,41,41))/40
        rc = self.midpoints(x)
        gc = self.midpoints(y)
        bc = self.midpoints(z)

        # combine the objects into a single boolean array
        voxels = mask ^ result_mask

        colors = np.zeros(voxels.shape + (3,))
        colors[..., 0] = rc
        colors[..., 1] = gc
        colors[..., 2] = bc

        # and plot everything
        fig = plt.figure()
        ax = fig.gca(projection='3d',alpha=0.3)
        ax.grid(False)

        # Hide axes ticks
        ax.set_xticks([])
        ax.set_yticks([])
        ax.set_zticks([])
        ax.voxels(voxels,facecolors=colors,  edgecolor="k",alpha=0.3)
        #ax.voxels(result_mask,  edgecolor="k",alpha=0.3)
        ax.view_init(elev=0,azim=800)
        plt.savefig("plots/plot3d.png")
    
    def plot_result4D(self, num=-1):
        result_mask, mask = self.sample_prediction()
        
        x, y, z = np.indices(np.array(mask.shape)+1)/config.shape[0]
        rc = self.midpoints(x)
        gc = self.midpoints(y)
        bc = self.midpoints(z)


        colors = np.zeros(mask.shape + (4,))
        colors[..., 0] = rc
        colors[..., 1] = gc
        colors[..., 2] = bc
        colors[..., 3] = 0.4

        frame_count = 100
        first_stage=frame_count//1.8


        def update_lines(num):#, dataLines, lines):
            rotation = 180 # degrees
            angle = rotation/first_stage
            if num == frame_count//1.5:
                voxelmap = ax.voxels(mask, facecolors=1-colors, edgecolor=None, label="Prediction")
            if num <= first_stage:
                ax.view_init(10,num*angle)
            else:
                ax.view_init(min(10+(num-first_stage)*angle,90),angle*first_stage)

        # Attaching 3D axis to the figure
        fig = plt.figure()
        ax = Axes3D(fig)

        voxelmap = ax.voxels(result_mask, facecolors=colors, edgecolor=None,alpha=0.3, label="Groundtruth")
        # Setting the axes properties


        ax.set_title('Aneurysm Segmentation')
        ax.grid(False)

        # Hide axes ticks
        ax.set_xticks([])
        ax.set_yticks([])
        ax.set_zticks([])

        # Creating the Animation object
        line_ani = animation.FuncAnimation(fig, update_lines, frame_count,#fargs={"voxelmap":voxelmap},
                                           interval=200, blit=False)


        Writer = animation.writers['ffmpeg']
        writer = Writer(fps=15)#, metadata=dict(artist='Me'), bitrate=5500)
        
        line_ani.save("plots/plot{}.mp4".format(num), writer=writer, dpi=200)

def load_custom_model(model_path):
    return load_model(model_path,custom_objects = {"GroupNormalization":norm.GroupNormalization, "loss":net.weighted_crossentropy(500), "dice_coef":net.dice_coef})
        
if __name__=="__main__":
    trainer = Trainer()
    
    unet = net.UNet(level_count=2, conv_count=2, loss=net.weighted_crossentropy(500), residual=True, filter_count=30,optimizer="adam")
    unet.build()
    
    trainer.train(epochs=30, model = unet.model, batch_size=2, descriptive_args=unet.get_args())
    #model_path = "~/Simon/src/aneurysm-segment/logs/20190623x064228TR/model.h5"
    #model = load_custom_model(config.model_path)
    
    #trainer.train(epochs=100, model = model, batch_size=2, descriptive_args="continuation of 20190623x064228TR")
    #trainer.plot_result4D(num=-3)
    #trainer.plot_result4D(num=-2)
    #trainer.plot_result4D(num=-5)
    