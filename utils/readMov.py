import numpy as np

def readMov(path, frames, height, width, barwidth, patch_edge_size=None, time_size = None, normalize_patch=False):
    """
    Reads in a movie chunk form the van Hatteren database.

    Parameters
    ----------
    path  :   string indicating filepath to read from 
    frames:   integer number of frames in the movie
    height:   integer height in pixels of each frame
    width:    integer width in pixels of each frame
    barwidth: integer height of black bar artifact in movie 
    patch_edge_size: integer length of patch in pixels
    
    Returns
    -------
    d       : numpy matrix with pixel values for movie.
    
    """
    with open(path, 'rb') as fid:
        #read in movie
        d = np.fromfile(fid, np.dtype('uint8'))

    #put data back into a movie shape
    d = np.reshape(d,(frames,height,width))
    #remove black bar from top, and make the other side even too
    d = d[:,barwidth:,int(barwidth/2):-int(barwidth/2)]
    #average each of 2 frames together
    d = np.mean(np.array([d[::2,],d[::2,]]),axis=0)
    tr_frames, tr_height, tr_width = np.shape(d)
    #print(np.shape(d))
    
    if (patch_edge_size != None):
        htiles = np.int(np.floor(tr_height/patch_edge_size))
        wtiles = np.int(np.floor(tr_width/patch_edge_size))
        ftiles = np.int(np.floor(tr_frames/time_size))
        #print('height'+str(tr_width)+'patch'+str(patch_edge_size)+'htiles'+str(htiles))
        print(np.shape(d))
        print('making patches...')
        tiled_d = np.asarray(np.split(d[:,:,0:patch_edge_size*wtiles], wtiles,2)) # tile column-wise
        print(np.shape(tiled_d))
        tiled_d = np.asarray(np.split(tiled_d[:,:,0:patch_edge_size*htiles], htiles,2)) #tile row-wise
        print(np.shape(tiled_d))
        tiled_d = np.asarray(np.split(tiled_d[:,:,0:time_size*ftiles], ftiles,2)) #tile time-wise
        print(np.shape(tiled_d))
        tiled_d = np.transpose(np.reshape(np.transpose(tiled_d,(4,5,3,0,1,2)),(patch_edge_size,patch_edge_size,time_size,-1)),(3,0,1,2)) #stack tiles together
        print(np.shape(tiled_d))
        if(normalize_patch):
            print('normalizing patches...')
            tiled_d = tiled_d - np.mean(tiled_d,axis=(1,2,3),keepdims=True)
            tiled_d = tiled_d/np.std(tiled_d,axis=(1,2,3),keepdims=True)
        return(tiled_d)
        
    else:
        return(d)