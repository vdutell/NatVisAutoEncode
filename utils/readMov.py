import numpy as np
import utils.conversions as cnv
import imageio


def readMovVH(path, frames, height, width, barwidth):
    """
    Reads in a movie chunk form the van Hatteren database.
    Parameters
    ----------
    path  :   string indicating filepath to read from 
    frames:   integer number of frames in the movie
    height:   integer height in pixels of each frame
    width:    integer width in pixels of each frame
    barwidth: integer height of black bar artifact in movie 
    
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
    tr_frames, tr_height, tr_width = np.shape(d)
    #print(np.shape(d))
    
    #video is in PAL format (interlaced?) take this into account by taking average of every other frame
    d = np.mean(np.array([d[::2,],d[1::2,]]),axis=0)

    return d


def readMovMp4(path):
    d = []
    reader = imageio.get_reader(path,'ffmpeg')
    #fps = reader.get_meta_data()['fps'] #we know this is 24
    for im in reader:
        d.append(im)
    d = np.array(d)[95:200]
    return d
        
    

def get_movie(movie_name, pixel_patch_size, frame_patch_size,
                          normalize_patch=False, normalize_movie=True):

    if(movie_name=='vh'):
        fpath = '/home/vasha/datasets/vanHaterenNaturalMovies/vid075'
        fps = 25 #approximated from http://redwood.berkeley.edu/bruno/data/vid075/README and increased by me.
        nframes = 9600
        rawframeh = 128
        rawframew = 128
        barw = 16
        framew = rawframew - barw #in pixels
        frameh = rawframeh - barw #in pixels

        #vhimgs, params['nimages'] = imr.check_n_load_ims(params['patchsize'], params['iterations'])
        m = rmov.readMovVH(fpath, nframes, rawframeh, rawframew, barw)
        
    if(movie_name=='cheetah'):
        fpath = '/home/vasha/datasets/cheetahlongclip.mp4'
        fps = 30
        ppd = 1./cnv.px2degfull(1)
        
        #read in movie
        m = readMovMp4(fpath)
        nframes, frameh, framew, ncolorchannels = np.shape(m)
        #tr_frames, tr_height, tr_width = np.shape(d)
    
        #remove color channel:
        m = np.mean(m,axis=3)
        print(m.shape)

    #convert to degrees
    framewdeg = framew/ppd 
    framehdeg = frameh/ppd
    
    #sampling rate
    deltawdeg = 1./ppd
    deltahdeg = 1./ppd 
    deltathz = 1./fps

    #normalize_movie
    if(normalize_movie):
        print('normalizing movie...')
        m = m - np.mean(m)
        m = m/np.std(m)
    
    #make patches
    if (pixel_patch_size != None):
        htiles = np.int(np.floor(frameh/pixel_patch_size))
        wtiles = np.int(np.floor(framew/pixel_patch_size))
        ftiles = np.int(np.floor(nframes/frame_patch_size))  
        
        print('making patches...')
        m = np.asarray(np.split(m[:,:,0:pixel_patch_size*wtiles], wtiles,2)) # tile column-wise
        m = np.asarray(np.split(m[:,:,0:pixel_patch_size*htiles], htiles,2)) #tile row-wise
        m = np.asarray(np.split(m[:,:,0:frame_patch_size*ftiles], ftiles,2)) #tile time-wise
        m = np.transpose(np.reshape(np.transpose(m,(4,5,3,0,1,2)),
                                   (pixel_patch_size, pixel_patch_size, frame_patch_size,-1)),(3,0,1,2)) #stack tiles together
        print(m.shape)
    #normalize patches
    if(normalize_patch):
        print('normalizing patches...')
        m = m - np.mean(m,axis=(1,2,3),keepdims=True)
        m = m/np.std(m,axis=(1,2,3),keepdims=True)
        
    #transpose & shuffle
    m = np.transpose(m, (0, 3, 1, 2)) #change axis to [batchsize, frame_patch_size, x_patchsize, y_patchsize]
    np.random.shuffle(m)
        
    return(m)

