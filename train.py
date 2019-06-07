

import tensorflow as tf
from tensorflow.keras.layers import Conv3D, Conv3DTranspose, Flatten, Dense
import numpy as np
import data
from config import patch_data_path
import preprocessing
import os
import net
import pandas as pd
from mpl_toolkits.mplot3d import Axes3D
import matplotlib.pyplot as plt
import matplotlib.animation as animation
import datetime
from tensorflow.keras.utils import plot_model

class Trainer:
    def __init__(self):
        os.makedirs("plots",exist_ok=True)
        os.makedirs("logs",exist_ok=True)
    
    def train(self, epochs=10, optimizer="adam", loss = "mean_squared_error", name="TR", model=None):
        self.loss=loss
        self.epochs = epochs
        self.optimizer = optimizer
        self.name = "{}{}".format(datetime.datetime.now().strftime("%Y%m%dx%H%M%S"),name)
        records = pd.read_csv(os.path.join(patch_data_path,"records.csv"))
        names = list(records[records["positiv"]]["filepath"])
        names = [name.split("/")[-1]+".npy" for name in names]

        self.adg = preprocessing.AneurysmDataGenerator(patch_data_path,names)

        

        self.x = self.adg.images
        self.x = np.array(self.x)
        self.x = self.x.reshape(*self.x.shape,1)
        self.y = self.adg.masks
        self.y = np.array(self.y)
        
        
        #if "crossentropy" in self.loss:
            
            #self.y = keras.utils.to_categorical(self.y)
            
            #self.model = net.conv_model_auto_deep(loss=loss, optimizer=optimizer, output_dim=2)
        #else:
            
        self.y = self.y.reshape(*self.y.shape,1)
            #self.model = net.conv_model_auto_deep(loss=loss, optimizer=optimizer)
        if model is None:
            self.model = net.unet(loss=loss, optimizer=optimizer)
        else:
            self.model=model

        self.history = self.model.fit(self.x,self.y,batch_size=20,epochs=epochs,validation_split=0.05)
        self.log()
        
    def log(self):
        local_path = "logs/{}".format(self.name)
        os.makedirs(local_path,exist_ok=True)
        pd.DataFrame(self.history.history).to_csv(os.path.join(local_path, "history.csv"))
        args = "{},{},{},{}".format(self.name,self.epochs, self.optimizer,self.loss)
        with open(os.path.join(local_path,"args.txt"),"w") as f:
            f.write(args)
            
        model_json = self.model.to_json()
        with open(os.path.join(local_path,"model.json"), "w") as json_file:
            json_file.write(model_json)
        self.model.save(os.path.join(local_path,'model.h5'))
        #plot_model(self.model, to_file=os.path.join(local_path,"model.png"))
        

    def sample_prediction(self, num=-1):
        result = self.model.predict(self.x[num:num+1])
        #if False and "crossentropy" in self.loss or True:
        #    return result[...,1].reshape((40,40,40))
        return result.reshape((40,40,40))
    
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
        result_mask = self.sample_prediction(num) > 0.40
        mask = self.adg.masks[num]
        
        x, y, z = np.indices(np.array(mask.shape)+1)/40
        rc = self.midpoints(x)
        gc = self.midpoints(y)
        bc = self.midpoints(z)


        colors = np.zeros(mask.shape + (4,))
        colors[..., 0] = rc
        colors[..., 1] = gc
        colors[..., 2] = bc
        colors[..., 3] = 0.3

        frame_count = 100
        first_stage=frame_count//1.8


        def update_lines(num):#, dataLines, lines):
            rotation = 180 # degrees
            angle = rotation/first_stage
            if num == frame_count//1.5:
                voxelmap = ax.voxels(mask, facecolors=1-colors, edgecolor="k", label="Prediction")
            if num <= first_stage:
                ax.view_init(10,num*angle)
            else:
                ax.view_init(min(10+(num-first_stage)*angle,90),angle*first_stage)

        # Attaching 3D axis to the figure
        fig = plt.figure()
        ax = Axes3D(fig)

        voxelmap = ax.voxels(result_mask, facecolors=colors, edgecolor="k",alpha=0.3, label="Groundtruth")
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
        
if __name__=="__main__":
    trainer = Trainer()
    #unet = net.UNet(level_count=1, conv_count=2, loss=net.weighted_crossentropy(32))
    unet = net.UNet(level_count=1, conv_count=2, loss=net.dice_coef_loss, residual=True)
    unet.build()
    
    trainer.train(epochs=50, loss=net.weighted_crossentropy(32), model = unet.model)
    #trainer.plot_result4D(num=-3)
    #trainer.plot_result4D(num=-2)
    #trainer.plot_result4D(num=-5)
    