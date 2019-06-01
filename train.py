import tensorflow as tf
import keras
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

class Trainer:
    def __init__(self):
        os.makedirs("plots",exist_ok=True)
    
    def train(self, epochs=10):
        records = pd.read_csv(os.path.join(patch_data_path,"records.csv"))
        names = list(records[records["positiv"]]["filepath"])
        names = [name.split("/")[-1]+".npy" for name in names]

        self.adg = preprocessing.AneurysmDataGenerator(patch_data_path,names)

        self.model = net.conv_model_auto_deep()

        self.x = self.adg.images#[image for i, image in enumerate(adg.images) if np.mean(adg.masks[i])>0.01]
        self.x = np.array(self.x)
        self.x = self.x.reshape(*self.x.shape,1)
        self.y = self.adg.images#[mask for i, image in enumerate(adg.masks) if np.mean(adg.masks[i])>0.01]
        self.y = np.array(self.y)
        self.y = self.y.reshape(*self.y.shape,1)

        self.model.fit(self.x,self.y,batch_size=20,epochs=epochs,validation_split=0.05)
        

    def sample_prediction(self):
        result = self.model.predict(self.x[-1:]).reshape((40,40,40))
        return result
    
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
    
    def plot_result4D(self):
        result_mask = self.sample_prediction() > 0.95
        mask = self.adg.masks[-1]
        
        x, y, z = np.indices((41,41,41))/40
        rc = self.midpoints(x)
        gc = self.midpoints(y)
        bc = self.midpoints(z)


        colors = np.zeros(mask.shape + (4,))
        colors[..., 0] = rc
        colors[..., 1] = gc
        colors[..., 2] = bc
        colors[..., 3] = 0.3

        frame_count = 100
        first_stage=frame_count//2


        def update_lines(num):#, dataLines, lines):
            angle = 90/first_stage
            if num == frame_count//1.5:
                voxelmap = ax.voxels(result_mask, facecolors=1-colors, edgecolor="k", label="Prediction")
            if num <= first_stage:
                ax.view_init(10,num*angle)
            else:
                ax.view_init(min(10+(num-first_stage)*angle,90),angle*first_stage)

        # Attaching 3D axis to the figure
        fig = plt.figure()
        ax = Axes3D(fig)

        voxelmap = ax.voxels(mask, facecolors=colors, edgecolor="k",alpha=0.3, label="Groundtruth")
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
        os.path
        line_ani.save("plots/plot.mp4", writer=writer, dpi=200)
        
if __name__=="__main__":
    trainer = Trainer()
    trainer.train()
    trainer.plot_result4D()