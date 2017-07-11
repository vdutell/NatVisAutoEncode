import numpy as np
import matplotlib.pyplot as plt
import matplotlib

"""
Author: Dylan Payton taken from FeedbackLCA code
Pad data with ones for visualization
Outputs:
  padded version of input
Args:
  data: np.ndarray
"""

def pad_data(data):
    n = int(np.ceil(np.sqrt(data.shape[0])))
    padding = (((0, n ** 2 - data.shape[0]),
    (1, 1), (1, 1))                       # add some space between filters
    + ((0, 0),) * (data.ndim - 3))        # don't pad the last dimension (if there is one)
    padded_data = np.pad(data, padding, mode="constant", constant_values=1)
    # tile the filters into an image
    padded_data = padded_data.reshape((n, n) + padded_data.shape[1:]).transpose((0, 2, 1, 3) + tuple(range(4, padded_data.ndim + 1)))
    padded_data = padded_data.reshape((n * padded_data.shape[1], n * padded_data.shape[3]) + padded_data.shape[4:])
    return padded_data


def normalize_data(data):
    norm_data = data.squeeze()
    if np.max(np.abs(data)) > 0:
        norm_data = (data / np.max(np.abs(data))).squeeze()
    return norm_data


"""
Author: Dylan Payton taken from FeedbackLCA code
Display input data as an image with reshaping
Outputs:
  fig: index for figure call
  sub_axis: index for subplot call
  axis_image: index for imshow call
Inpus:
  data: np.ndarray of shape (height, width) or (n, height, width)
  normalize: [bool] indicating whether the data should be streched (normalized)
    This is recommended for dictionary plotting.
  title: string for title of figure
"""
def display_data_tiled(data, normalize=False, title="", prev_fig=None):
       
    #calculate mean of each picture of weights
    mean_list =[]
    for x in data:
        mean_list.append(np.linalg.norm(np.reshape(x,-1),ord=2))
        #mean_list.append(np.linalg.norm(np.reshape(x,-1)))
    
    #Rescale data    
    mean_data = np.mean(data)
    min_data = np.amin(data)
    max_data = np.amax(data)
    data = (((data-min_data)/(max_data-min_data))*2)-1
    
    if normalize:
        data = normalize_data(data)
    if len(data.shape) >= 3:
        data = pad_data(data)
        
    fig = plt.figure(figsize=(10,10))
    
    sub_axis = fig.add_subplot(2,1,1)  
    axis_image = sub_axis.imshow(data, 
                                 cmap="Greys_r",
                                 interpolation="none")
    axis_image.set_clim(vmin=-1.0, vmax=1.0)
    # Turn off tick labels
    sub_axis.set_yticklabels([])
    sub_axis.set_xticklabels([])
    cbar = fig.colorbar(axis_image)
    sub_axis.tick_params(
        axis="both",
        bottom="off",
        top="off",
        left="off",
        right="off")  
    
    bar_chart = fig.add_subplot(2,1,2)
    bar_chart.bar(range(0, len(mean_list)), mean_list, edgecolor = 'black', color = 'black')

    #fig.subtitle(title, y=1.05)
    fig.canvas.draw()
    #plt.show()
    
    return (fig, sub_axis, axis_image)


"""
Author: Vasha DuTell
Plot to visualize the tiling of the center RF of on and off cells separately.
Outputs:
  Figure object with two tiling plots, one with on, and the other with off cells.
Args:
  data: np.ndarray or list of weights, each an individiaul neuron RF
"""
def plotonoff(allws):

    #extract on center
    onws = np.mean(allws,axis=0)>0
    onws = allws[:,onws]
    #extract off center
    offws = np.mean(allws,axis=0)<0
    offws = allws[:,offws]
    #keep track of the circles
    oncircs = []
    offcircs = []

    for ws in allws:
        circ = (ws>(0.9*np.sign(np.mean(ws))))
        if(np.mean(ws)>0):
            oncircs.append(circ)
        else:
            offcircs.append(False==circ)

    #plot
    fig = plt.figure(figsize=(6,3.5))
    plt.subplot(1,2,1,title='On')    
    oncolors = iter(plt.cm.jet(np.linspace(0,1,len(oncircs))))           
    for onc in oncircs: 
        plt.contour(onc,[0.7],linewidths = 3,colors=[next(oncolors)])
    plt.xticks([])
    plt.yticks([])
    
    plt.subplot(1,2,2,title='Off')
    offcolors = iter(plt.cm.jet(np.linspace(0,1,len(offcircs))))  
    for ofc in offcircs:
        plt.contour(ofc,[0.7], linewidths = 3, colors=[next(offcolors)])
    plt.xticks([])
    plt.yticks([])
    plt.tight_layout()
    
    return(fig)




def save_plots(aec,
               activations,
               cost_evolution,
                wmean_evolution,
                inweights_evolution,
                outweights_evolution,
                images,
                recons,
                final_inweights,
                final_outweights):
    
    savefolder = aec.params['savefolder']

    #Save our final weights
    inweights_evolution_r = np.rollaxis(np.reshape(inweights_evolution,
                                                 (len(inweights_evolution),
                                                  aec.params['imxlen'],
                                                  aec.params['imylen'],
                                                  aec.params['nneurons'])),3,1)
    (f,sa,ai) = display_data_tiled(inweights_evolution_r[-1], normalize=True, title="final_in_weights", prev_fig=None);
    f.savefig(savefolder+'inweights_final.png')
    plt.close()    
    
    outweights_evolution_r = np.reshape(outweights_evolution,
                                         (len(outweights_evolution),
                                          aec.params['nneurons'],
                                          aec.params['imxlen'],
                                          aec.params['imylen'])) #no rollaxis needed b/c shape is already nnuerons in pos 1.
    
    (f,sa,ai) = display_data_tiled(outweights_evolution_r[-1], normalize=True, title="final_out_weights", prev_fig=None);
    f.savefig(savefolder+'outweights_final.png')
    plt.close()

    #save evolving weights
    for i in range(len(inweights_evolution_r)):
        (f,sa,ai) = display_data_tiled(inweights_evolution_r[i], normalize=True,title="inweights_evolving", prev_fig=None);
        f.savefig(savefolder+'/inweights_evolution_'+str(i)+'.png')
        plt.close()
        
        (f,sa,ai) = display_data_tiled(outweights_evolution_r[i], normalize=True,title="outweights_evolving", prev_fig=None);
        f.savefig(savefolder+'/outweights_evolution_'+str(i)+'.png')
        plt.close()
        
    #save plot of activations
    f8 = plt.figure(figsize=(6,6))
    plt.plot(activations)
    plt.title('Activations')
    f8.savefig(savefolder+'/activations.png') 
    plt.close()
    
    #save weights and cost evolution
    f2 = plt.figure(figsize=(6,6))
    plt.subplot(2,1,1,title='Weights_Mean')
    plt.plot(wmean_evolution)
    plt.subplot(2,1,2,title='Cost')
    plt.plot(cost_evolution)
    #plt.plot(cost_evolution/2)
    plt.tight_layout()
    f2.savefig(savefolder+'/cost_weights.png') 
    plt.close()
    
    #show an example image and reconstruction from the last iteration of learning
    patchnum = 3
    plots = 4
    f3 = plt.figure()
    for i in range(plots):
        plt.subplot(plots,2,2*i+1)#,title='Patch')
        plt.imshow(images[-1][patchnum+i],cmap='gray',interpolation='none')
        plt.colorbar()
        plt.axis('off')

        plt.subplot(plots,2,2*i+2)#,title='Recon')
        plt.imshow(recons[-1][patchnum+i],cmap='gray',interpolation='none')
        plt.colorbar()
        plt.axis('off')

    plt.tight_layout()
    f3.savefig(savefolder+'/reconstruction.png') 
    plt.close() 
    
    #save plots of on and off tiling
    f4 = plotonoff(inweights_evolution_r[-1]);
    f4.savefig(savefolder+'/final_in_on_off_RFs.png') 
    plt.close()
    
    #save plots of on and off tiling
    f5 = plotonoff(outweights_evolution_r[-1]);
    f5.savefig(savefolder+'/final_out_on_off_RFs.png') 
    plt.close()


