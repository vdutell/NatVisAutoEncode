import numpy as np
import tensorflow as tf

def fr2ms(frames, fps):
    ms = (float(frames) * 1000)/fps
    return(ms)

def ms2fr(ms, fps):
    frames = (float(ms)*fps)/1000
    frames = int(np.round(frames,decimals=0))
    if(frames==0):
        frames=1
    return(frames)

def px2degfull(px):
    """
    Converts pixels on my 15" macbook pro at 18" viewing dist to degrees on retina
    Usefull in converting cycles per pixel -> cycles per degree
    
    Parameters
    ----------
    px: integer pixels
 
    Returns
    -------
    degrees : float value for degrees
    
    """
    d = 18 #viewing distance in inches
    ppi = 227 #my macbook - other 132.07 #pixels per inch
    inches = px/ppi
    
    visang = 2 * (180/np.pi) * np.arctan(inches/2/d)
    
    return(visang)

#transform weight tensor to image format
def transform_weight_to_image(weight_vec, params):
    image = tf.transpose(weight_vec, perm=[1,2,4,0,3])
    image = tf.reshape(image, (params["pixelpatchsize"], params["pixelpatchsize"], params['nneurons']))#,1))
    return image

def some_conversions():
    #convert to degrees
    ppd = 6 #pixels per degree subtended on retina (estimated 10deg for 64px in dong atick 95)
    framewdeg = framew/ppd 
    framehdeg = frameh/ppd
    #sampling rate
    deltawdeg = 1./ppd
    deltahdeg = 1./ppd 
    deltathz = 1./fps