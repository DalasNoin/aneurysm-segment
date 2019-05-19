from matplotlib import pyplot as plt

def plot_overlap(target_mask, predicted_mask):
    import matplotlib.pyplot as plt
    N=4
    image = target_mask
    mask = predicted_mask > 0.65
    plt.figure(dpi=800)
    plt.rcParams['figure.figsize'] = [20,20]
    fig, ax = plt.subplots(nrows=N, ncols=N)
    for i in range(N):
        for j in range(N):
            im = image[i+j*N+5,:,:]
            m = mask[i+j*N+5,:,:]
            ax[i,j].imshow(im,alpha=0.4,cmap="Greys_r",vmin=0,vmax=1)
            ax[i,j].imshow(m,alpha=0.6, cmap="cividis",vmin=0,vmax=1)
    return fig