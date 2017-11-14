import numpy as np
import matplotlib.pyplot as plt
import matplotlib

from matplotlib.ticker import NullFormatter
from sklearn import manifold
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

    #Rescale data    
    mean_data = np.mean(allws)
    min_data = np.amin(allws)
    max_data = np.amax(allws)
    data = (((allws-min_data)/(max_data-min_data))*2)-1
    #data = normalize_data(data)
    
    #extract on center
    onws = np.mean(allws,axis=0)>0
    onws = allws[:,onws]
    #extract off center
    offws = np.mean(allws,axis=0)<0
    offws = allws[:,offws]
    #keep track of the circles
    oncircs = []
    offcircs = []
    ambiguous = []
    labels = []

    circthresh = 0.5
    onoffthresh = 0
    
    for ws in allws:
        if(np.mean(ws)>onoffthresh):
            circ = (ws>(circthresh*np.sign(np.mean(ws))))
            oncircs.append(circ)
            labels.append(1)
        elif(np.mean(ws)<-onoffthresh):
            circ = (ws<(circthresh*np.sign(np.mean(ws))))
            offcircs.append(circ)
            labels.append(-1)
        else:
            labels.append(0)

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
    
    return(labels, fig)

def measure_plot_dist(weight_mat, norm, normalize=True):
    ## measures pairwise norm of hidden node weights.
    ## Inputs:
    ## weight_mat: matrix of weights of shape nneurons by input shape (input shape can be 1 or 2d)
    ## norm: String describing the type of norm to take - see acceptible norms in documentation for scipy.spatial.distance.pdist
    ## normalize: boolean whether to normalize weight vectors to unit norm
    
    ## Outputs:
    ## dist: a nneurons by nneurons matrix, with pairwise distances of the weight matrices in each element
    ## fig: plot of the distance matrix as a heatmap

    #vectorize
    fwv = weight_mat.reshape(weight_mat.shape[0],-1)
    
    #make each weigth vector unit norm
    if(normalize):
        fwv = fwv / np.linalg.norm(fwv,axis=0)
        
    dist = scpd.pdist(fwv,norm) #'euclidean','hamming'
    dist = scpd.squareform(dist)
    
    fig = plt.figure(figsize=(6,6))
    plt.imshow(dist)
    plt.colorbar()

    return(dist, fig)


def measure_plot_act_corrs(activations):
    ccf = np.corrcoef(np.array(activations).T)
    fig = plt.figure(figsize = (6,6))
    plt.imshow(ccf)
    plt.colorbar()
    
    return(ccf,fig)


def plot_dist_embeddings(distmat, onofflabels, n_neighbors = 10, n_components = 2):
     
    fig = plt.figure(figsize = (6,6))

    #isomap
    iso = manifold.Isomap(n_neighbors, n_components).fit_transform(distmat)
    ax = fig.add_subplot(2, 2, 1)
    plt.scatter(iso[:, 0], iso[:, 1], c=onofflabels)
    plt.title('Isomap - {} Neighbors'.format(n_neighbors))
    ax.xaxis.set_major_formatter(NullFormatter())
    ax.yaxis.set_major_formatter(NullFormatter())
    plt.axis('tight')

    #Spectral
    spec = manifold.SpectralEmbedding(n_components=n_components, n_neighbors = n_neighbors).fit_transform(distmat)
    ax = fig.add_subplot(2, 2, 2)
    plt.scatter(spec[:, 0], spec[:, 1], c=onofflabels)
    plt.title('Spectral - {} Neighbors'.format(n_neighbors))
    ax.xaxis.set_major_formatter(NullFormatter())
    ax.yaxis.set_major_formatter(NullFormatter())
    plt.axis('tight')

    #TSNE
    tsne = manifold.TSNE(n_components, init='pca', random_state=0).fit_transform(distmat)
    ax = fig.add_subplot(2, 2, 3)
    plt.scatter(tsne[:, 0], tsne[:, 1], c=onofflabels)
    plt.title('t-SNE - {} Components'.format(n_components))
    ax.xaxis.set_major_formatter(NullFormatter())
    ax.yaxis.set_major_formatter(NullFormatter())
    plt.axis('tight')

    #MDS
    max_iter = 10
    mds = manifold.MDS(n_components=n_components, metric=True, max_iter=max_iter, dissimilarity="precomputed").fit_transform(distmat)
    ax = fig.add_subplot(2, 2, 4)
    plt.scatter(mds[:, 0], mds[:, 1], c=onofflabels)
    plt.title('MDS - {} Components'.format(n_components))
    ax.xaxis.set_major_formatter(NullFormatter())
    ax.yaxis.set_major_formatter(NullFormatter())
    plt.axis('tight')

    return(fig)


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
    f1 = plt.figure(figsize=(6,6))
    plt.plot(final_acts)
    plt.title('Activations')
    f1.savefig(savefolder+'/trained_activations.png') 
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
    onofflabels, f4 = plotonoff(fiw);
    f4.savefig(savefolder+'/trained_in_on_off_RFs.png') 
    plt.close()
    
    #save distance plots
    dists, f5 = measure_plot_dist(fiw, norm='euclidean');
    f5.savefig(savefolder+'/trained_distances.png') 
    plt.close()
        
    #save plots of clustering
    f6 = plot_dist_embeddings(dists, onofflabels, n_neighbors=5)
    f6.savefig(savefolder+'/trained_manifold_embeddings_RFs.png') 
    plt.close()
    
    #save activation correlation plots
    corrs, f7 =  measure_plot_act_corrs(activations);
    f7.savefig(savefolder+'/trained_act_corrs.png') 
    plt.close()
    
    #save plots of activation
    for i in range(len(activation_evolution)):
        f8 = plt.figure()
        plt.bar(range(0, len(activation_evolution[i])), activation_evolution[i], edgecolor = 'black', color = 'black')
        f8.savefig(savefolder+'param_evolution/activation_'+str(i)+'.png')
        plt.close()
        
    #save plots of inbiases
    for i in range(len(inbias_evolution)):
        f9 = plt.figure()
        plt.bar(range(0, len(inbias_evolution[i])), inbias_evolution[i], edgecolor = 'black', color = 'black')
        f9.savefig(savefolder+'param_evolution/inbias_'+str(i)+'.png')
        plt.close()
        
        