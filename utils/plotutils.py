import numpy as np
import matplotlib.pyplot as plt
import matplotlib
    
import scipy.spatial.distance as scpd
    
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
def display_data_tiled(data, acts, normalize=False, title="", prev_fig=None):
       
    #calculate mean of each picture of weights
    mean_list =[]
    for x in data:
        mean_list.append(np.linalg.norm(np.reshape(x,-1),ord=2))
        #mean_list.append(np.linalg.norm(np.reshape(x,-1)))
    
    mean_list = np.array(mean_list)
    
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
    bar_chart.bar(range(0, len(acts)), acts, edgecolor = 'black', color = 'black')

    #bar_chart.title()
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
        circ = (ws>(0.3*np.sign(np.mean(ws))))
        if(np.mean(ws)>0):
            oncircs.append(circ)
        else:
            offcircs.append(False==circ)

    #plot
    fig = plt.figure(figsize=(6,3.5))
    plt.subplot(1,2,1,title='On')    
    oncolors = iter(plt.cm.jet(np.linspace(0,1,len(oncircs))))           
    for onc in oncircs: 
        plt.contour(onc,[0.3],linewidths = 3,colors=[next(oncolors)])
    plt.xticks([])
    plt.yticks([])
    
    plt.subplot(1,2,2,title='Off')
    offcolors = iter(plt.cm.jet(np.linspace(0,1,len(offcircs))))  
    for ofc in offcircs:
        plt.contour(ofc,[0.3], linewidths = 3, colors=[next(offcolors)])
    plt.xticks([])
    plt.yticks([])
    plt.tight_layout()
    
    return(fig)

def measure_plot_dist(weight_mat,norm,plot=True):
    ## measures pairwise norm of hidden node weights.
    ## Inputs:
    ## weight_mat: matrix of weights of shape nneurons by input shape (input shape can be 1 or 2d)
    ## norm: String describing the type of norm to take - see acceptible norms in documentation for scipy.spatial.distance.pdist
    ## plot: Boolean indicating whether or not to plot the heatmap of the weight matrix disances.
    
    ## Outputs:
    ## dist: a nneurons by nneurons matrix, with pairwise distances of the weight matrices in each element
    ## optional plot of the distance matrix as a heatmap

    fwv = weight_mat.reshape(weight_mat.shape[0],-1)
    dist = scpd.pdist(fwv,norm) #'euclidean','hamming'
    dist = scpd.squareform(dist)
    if(plot==True):
        plt.imshow(dist)
        plt.colorbar()
    return(dist)


def save_plots(aec,
               activations,
               cost_evolution,
               wmean_evolution,
               inweights_evolution,
               outweights_evolution,
               images,
               recons,
               final_inweights_ordered,
               final_outweights_ordered,
               inbias_evolution,
               activation_evolution):
    
    savefolder = aec.params['savefolder']

    #Save our final weights
    ## in weights
    fiw = final_inweights_ordered.reshape(aec.params['imxlen'],
                                  aec.params['imylen'],
                                  aec.params['nneurons']).T
    final_acts = activations[-1]
    (f,sa,ai) = display_data_tiled(fiw, final_acts[np.argsort(-final_acts)], normalize=True, title="final_in_weights", prev_fig=None);
    f.savefig(savefolder+'trained_weights_in.png')
    plt.close()    
    
    ##out weights
    fow = final_outweights_ordered.reshape(aec.params['imxlen'],
                                  aec.params['imylen'],
                                  aec.params['nneurons']).T
    (f,sa,ai) = display_data_tiled(fow, final_acts[np.argsort(-final_acts)], normalize=True, title="final_out_weights", prev_fig=None);
    
    f.savefig(savefolder+'trained_weights_out.png')
    plt.close()
   
    #save evolving weights
    inweights_evolution_r = np.rollaxis(np.reshape(inweights_evolution,
                                         (len(inweights_evolution),
                                          aec.params['imxlen'],
                                          aec.params['imylen'],
                                          aec.params['nneurons'])),3,1)
    outweights_evolution_r = np.reshape(outweights_evolution,
                                         (len(outweights_evolution),
                                          aec.params['nneurons'],
                                          aec.params['imxlen'],
                                          aec.params['imylen'])) #no rollaxis needed b/c shape is already nnuerons in pos 1.    
  
    for i in range(len(inweights_evolution_r)):
        (f,sa,ai) = display_data_tiled(inweights_evolution_r[i], activations[-1], normalize=True,title="inweights_evolving", prev_fig=None);
        f.savefig(savefolder+'param_evolution/inweights_evolution_'+str(i)+'.png')
        plt.close()
        
        (f,sa,ai) = display_data_tiled(outweights_evolution_r[i], activations[-1], normalize=True,title="outweights_evolving", prev_fig=None);
        f.savefig(savefolder+'param_evolution/outweights_evolution_'+str(i)+'.png')
        plt.close()
        
    #save plot of activations
    f8 = plt.figure(figsize=(6,6))
    plt.plot(final_acts)
    plt.title('Activations')
    f8.savefig(savefolder+'/param_evolution/trained_activations.png') 
    plt.close()
    
    #save weights and cost evolution
    f2 = plt.figure(figsize=(6,6))
    plt.subplot(2,1,1,title='Weights_Mean')
    plt.plot(wmean_evolution)
    plt.subplot(2,1,2,title='Cost')
    plt.plot(cost_evolution)
    plt.tight_layout()
    f2.savefig(savefolder+'/summary_weights_cost.png') 
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
    f4.savefig(savefolder+'/trained_in_on_off_RFs.png') 
    plt.close()
    
    #save plots of on and off tiling
    f5 = plotonoff(outweights_evolution_r[-1]);
    f5.savefig(savefolder+'/trained_out_on_off_RFs.png') 
    plt.close()
    
    
    #save plots of activation
    for i in range(len(activation_evolution)):
        f6 = plt.figure()
        plt.bar(range(0, len(activation_evolution[i])), activation_evolution[i], edgecolor = 'black', color = 'black')
        f6.savefig(savefolder+'param_evolution/activation_'+str(i)+'.png')
        plt.close()
    for i in range(len(inbias_evolution)):
        f9 = plt.figure()
        plt.bar(range(0, len(inbias_evolution[i])), inbias_evolution[i], edgecolor = 'black', color = 'black')
        f9.savefig(savefolder+'param_evolution/inbias_'+str(i)+'.png')
        plt.close()
        
        