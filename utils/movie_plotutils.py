import numpy as np
import matplotlib.pyplot as plt
import matplotlib
import matplotlib.animation as animation

"""
Author: Dylan Payton taken from FeedbackLCA code
Pad data with ones for visualization
Outputs:
  padded version of input
Args:
  data: np.ndarray
"""

#take the normalized weight matrix and reformate for plotting
def pad_data(data_full):
    
    padded_data_full = np.empty((0,96,96)) #space needed for plotting (2+10)*8
    
    for data in data_full:
        n = int(np.ceil(np.sqrt(data.shape[0])))
        padding = (((0, n ** 2 - data.shape[0]),
        (1, 1), (1, 1))                       # add some space between filters
        + ((0, 0),) * (data.ndim - 3))        # don't pad the last dimension (if there is one)
        padded_data = np.pad(data, padding, mode="constant", constant_values=1)
        # tile the filters into an image
        padded_data = padded_data.reshape((n, n) + padded_data.shape[1:]).transpose((0, 2, 1, 3) + tuple(range(4, padded_data.ndim + 1)))
        padded_data = padded_data.reshape((n * padded_data.shape[1], n * padded_data.shape[3]) + padded_data.shape[4:])
        
        padded_data_full = np.append(padded_data_full, [padded_data], axis=0)
        
        #print (type(padded_data))
        
    #padded_data_full = list(padded_data_full)    
    #print (type(padded_data_full))
             
    return padded_data_full



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

#calculate mean and data (frames of kernel) from weight matrix
def calculate_data_to_plot(data, normalize=False):

    #calculate mean of each picture of weights
    mean_list =[]
    for t in range(len(data)):
        mean_list.append([])
        for x in data[t]:
            mean_list[t].append(np.mean(np.absolute(x)))
        
    
    
    if normalize:
        data = normalize_data(data)
    else:
        #Rescale data    
        mean_data = np.mean(data)
        min_data = np.amin(data)
        max_data = np.amax(data)
        #print ('M=', mean_data)
        #print ('min_data=', min_data)
        #print ('max_data=', max_data)
        data = (((data-min_data)/(max_data-min_data))*2)-1
        
    if len(data.shape) >= 3:
        data = pad_data(data)
    
    return (data, mean_list)


def display_data_tiled(data, normalize=False, title="", prev_fig=None):
    
    data, mean_list = calculate_data_to_plot(data)
                
    fig = plt.figure() #figsize=(10,10))
    
    #print (data)
    #print (np.shape(data))
    
    sub_axis = fig.add_subplot(2,2,1)  
    axis_image = plt.imshow(data[0,:,:], cmap="Greys_r", interpolation="none", animated=True)
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
    
    bar_chart = fig.add_subplot(2,2,3)
    #print (len(mean_list))
    bar_chart.bar(range(0, len(mean_list[0])), mean_list[0], edgecolor = 'black', color = 'black')

    fig.suptitle(title, y=1.05)
    fig.canvas.draw()   

    
    def updatefig(i):
        axis_image.set_array(data[i,:,:])
        return axis_image,

    ani = animation.FuncAnimation(fig, updatefig, frames=range(len(data)), interval=50, blit=False)
    
    return ani

"""
def display_data_tiled(data, normalize=False, title="", prev_fig=None):
 
    #calculate mean of each picture of weights
    mean_list =[]
    for x in data:
        mean_list.append(np.mean(np.absolute(x)))
        
    #Rescale data    
    mean_data = np.mean(data)
    min_data = np.amin(data)
    max_data = np.amax(data)
    #print ('M=', mean_data)
    #print ('min_data=', min_data)
    #print ('max_data=', max_data)
    data = (((data-min_data)/(max_data-min_data))*2)-1
    
    if normalize:
        data = normalize_data(data)
    if len(data.shape) >= 3:
        data = pad_data(data)
        
    fig = plt.figure() #figsize=(10,10))
    
    sub_axis = fig.add_subplot(2,2,1)  
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
    
    bar_chart = fig.add_subplot(2,2,3)
    bar_chart.bar(range(0, len(mean_list)), mean_list, edgecolor = 'black', color = 'black')

    fig.suptitle(title, y=1.05)
    fig.canvas.draw()
    #plt.show()
    
    return (fig, sub_axis, axis_image)
"""


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
        circ = (ws>(0.99*np.sign(np.mean(ws))))
        if(np.mean(ws)>0):
            oncircs.append(circ)
        else:
            offcircs.append(False==circ)

    #plot
    fig = plt.figure(figsize=(6,3.5))
    plt.subplot(1,2,1,title='On')    
    oncolors = iter(plt.cm.jet(np.linspace(0,1,len(oncircs))))           
    for onc in oncircs: 
        plt.contour(onc,[0.99],linewidths = 3,colors=[next(oncolors)])
    plt.xticks([])
    plt.yticks([])
    
    plt.subplot(1,2,2,title='Off')
    offcolors = iter(plt.cm.jet(np.linspace(0,1,len(offcircs))))  
    for ofc in offcircs:
        plt.contour(ofc,[.99], linewidths = 3, colors=[next(offcolors)])
    plt.xticks([])
    plt.yticks([])
    plt.tight_layout()
    
    return(fig)




def save_plots(aec,
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
                                                  aec.params['frames_per_channel'],
                                                  aec.params['imxlen'],
                                                  aec.params['imylen'],
                                                  aec.params['nneurons'])),4,2)
    #(f,sa,ai) = display_data_tiled(inweights_evolution_r[-1], normalize=False, title="final_in_weights", prev_fig=None);
    #f.savefig(savefolder+'inweights_final.png')    
    ani = display_data_tiled(inweights_evolution_r[-1], normalize=False, title="final_in_weights", prev_fig=None);
    ani.save(savefolder+'inweights_final.mp4')
    plt.close()    
    
    outweights_evolution_r = np.rollaxis(np.reshape(outweights_evolution,
                                                  (len(inweights_evolution),
                                                   aec.params['frames_per_channel'],
                                                   aec.params['imxlen'],
                                                   aec.params['imylen'],
                                                   aec.params['nneurons'])),4,2)
    
    #(f,sa,ai) = display_data_tiled(outweights_evolution_r[-1], normalize=False, title="final_out_weights", prev_fig=None);
    #f.savefig(savefolder+'outweights_final.png')
    ani = display_data_tiled(outweights_evolution_r[-1], normalize=False, title="final_out_weights", prev_fig=None);
    ani.save(savefolder+'outweights_final.mp4')
    plt.close()

    #save evolving weights
    for i in range(len(inweights_evolution_r)):
        #(f,sa,ai) = display_data_tiled(inweights_evolution_r[i], normalize=False,title="inweights_evolving", prev_fig=None);
        #f.savefig(savefolder+'/inweights_evolution_'+str(i)+'.png')
        ani = display_data_tiled(inweights_evolution_r[i], normalize=False,title="inweights_evolving", prev_fig=None);
        ani.save(savefolder+'/inweights_evolution_'+str(i)+'.mp4')
        plt.close()
        
        #(f,sa,ai) = display_data_tiled(outweights_evolution_r[i], normalize=False,title="outweights_evolving", prev_fig=None);
        #f.savefig(savefolder+'/outweights_evolution_'+str(i)+'.png')
        ani = display_data_tiled(outweights_evolution_r[i], normalize=False,title="outweights_evolving", prev_fig=None);
        ani.save(savefolder+'/outweights_evolution_'+str(i)+'.mp4')      
        plt.close()
        
        
      
    #save weights and cost evolution
    f2 = plt.figure(figsize=(6,6))
    plt.subplot(2,1,1,title='Weights_Mean')
    plt.plot(wmean_evolution)
    plt.subplot(2,1,2,title='Objective')
    plt.plot(cost_evolution)
    plt.tight_layout()
    f2.savefig(savefolder+'/cost_weights.png') 
    plt.close()
    
    #show an example image and reconstruction 
    patchnum = 3
    plots = 4
    f3 = plt.figure()
    for i in range(len(inweights_evolution_r)):
        for j in range(plots):
            plt.subplot(plots,2,2*j+1)#,title='Patch')
            plt.imshow(images[i][patchnum+j],cmap='gray',interpolation='none')
            plt.axis('off')

            plt.subplot(plots,2,2*j+2)#,title='Recon')
            plt.imshow(recons[i][patchnum+j],cmap='gray',interpolation='none')
            plt.axis('off')

        plt.tight_layout()
        f3.savefig(savefolder+'/reconstruction_'+str(i)+'.png') 
    plt.close() 
    
    #save plots of on and off tiling
    #f4 = plotonoff(inweights_evolution_r[-1]);
    #f4.savefig(savefolder+'/final_in_on_off_RFs.png') 
    #plt.close()
    
    #save plots of on and off tiling
    #f5 = plotonoff(outweights_evolution_r[-1]);
    #f5.savefig(savefolder+'/final_out_on_off_RFs.png') 
    #plt.close()