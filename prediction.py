from tensorflow import keras
from tensorflow.keras import losses

import numpy as np
from data import patching, sequence_generators
import config
from tensorflow.keras.models import load_model
import preprocessing

import net
from mpl_toolkits.mplot3d import Axes3D
import matplotlib.pyplot as plt
import matplotlib.animation as animation



class Prediction:
    def __init__(self,model_path=config.model_path, do_load_model=True):
        if do_load_model:
            self.model = load_model(model_path, custom_objects={"loss":net.weighted_crossentropy(500), "dice_coef":net.dice_coef, "dice_coef_loss":net.dice_coef_loss})
        
    def forward_from_path(self, path):
        image = preprocessing.extract_pixel_array(path)
        image = (image - config.mean)/config.std
        return self.forward(image)

    def forward(self, tensor):
        shape = tensor.shape
        output = np.zeros(shape)
        pi = patching.PatchImage(name="input", tensor=tensor, shape=config.default_shape, stride=config.default_shape)
        
        for patch_tensor, indices, patch in pi.iterate():
            tensor = np.zeros((1,*config.shape,1))
            patch_shape = patch_tensor.shape
            tensor[0, :patch_shape[0], :patch_shape[1], :patch_shape[2],:] = patch_tensor.reshape(*patch_shape, 1)
            
            output[patch[0]:patch[3], patch[1]:patch[4], patch[2]:patch[5]] = self.model.predict(tensor)[0, :patch_shape[0], :patch_shape[1], :patch_shape[2],:].reshape(*patch_shape)

        return output
    
    def midpoints(self, x):
        sl = ()
        for i in range(x.ndim):
            x = (x[sl + np.index_exp[:-1]] + x[sl + np.index_exp[1:]]) / 2.0
            sl += np.index_exp[:]
        return x
    
    def plot_result4D(self, result_mask, mask, frame_count=100,name="mask"):
        
        x, y, z = np.indices(np.array(mask.shape)+1)/(mask.shape[1]+1)
        rc = self.midpoints(x)
        gc = self.midpoints(y)
        bc = self.midpoints(z)


        colors = np.zeros(mask.shape + (4,))
        colors[..., 0] = rc
        colors[..., 1] = gc
        colors[..., 2] = bc
        colors[..., 3] = 0.4

        
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
        
        line_ani.save("plots/plot{}.mp4".format("_example4D"+name), writer=writer, dpi=200)
        
    def plot_blood_result4D(self, result_mask, mask, frame_count=100,name="blood"):
        
        colors = ( 0.541,0.012, 0.012, 0.8)

        
        first_stage=frame_count//1.8


        def update_lines(num):#, dataLines, lines):
            rotation = 180 # degrees
            angle = rotation/first_stage
            if num == frame_count//1.5:
                fig = plt.figure()
                ax = Axes3D(fig)

                voxelmap = ax.voxels(mask, facecolors=colors, edgecolor=None, label="Prediction")                
                ax.set_title('Aneurysm Segmentation')
                ax.grid(False)

                # Hide axes ticks
                ax.set_xticks([])
                ax.set_yticks([])
                ax.set_zticks([])
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
        
        line_ani.save("plots/plot{}.mp4".format("_example4D"+name), writer=writer, dpi=200)
        
def save_plot(tensor, name):
    from scipy.ndimage import zoom
    fig = plt.figure()
    ax = fig.gca(projection='3d',alpha=0.3)
    ax.grid(False)
    ax.voxels(zoom(tensor,0.3))
    plt.savefig(f"plots/validation_out_{name}.pdf")
    

def evaluate_model(model_path):
    model = load_model(model_path,custom_objects={"loss":net.weighted_crossentropy(500),"dice_coef":net.dice_coef})

    partition = sequence_generators.get_train_val_sequence()
    test_gen = sequence_generators.DataGenerator(partition["test"],config.patch_validation_data_path, batch_size=2)
    print(model.evaluate_generator(test_gen))
    print(model.metrics_names)
    
if __name__=="__main__":
    evaluate_model(config.model_path)
    