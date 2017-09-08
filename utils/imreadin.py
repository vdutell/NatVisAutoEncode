import os
import numpy as np
import h5py
import glob
from scipy import io

class imageFile:
    def __init__(self,
                 img_dir,
                 patch_edge_size=None,
                 normalize_im=False,
                 normalize_patch=False,
                 invert_colors=False,
                 rand_state=np.random.RandomState()):
 
        # readin images
        self.images = self.extract_images(img_dir)    
        # process images
        self.images = self.process_images(self.images, patch_edge_size, 
                                          normalize_im, normalize_patch, invert_colors)

    def extract_images(self, img_dir):
        if('.h5' in img_dir):
            with h5py.File(img_dir, "r") as f:
                full_img_data = np.array(f['van_hateren_good'], dtype=np.float32) 
        elif('.mat' in img_dir):
            bw_ims = []
            for file in glob.glob(img_dir,recursive=True):
                mat = io.loadmat(file)
                #short medium and long activations
                sml_acts = np.array([mat['OS'],mat['OM'],mat['OL']])
                #mean over three for luminance
                bw_acts = np.mean(sml_acts,axis=0)
                # transpose if we need to ***I think this is allowed***
                if(np.shape(bw_acts)[0] > np.shape(bw_acts)[1]):
                    bw_acts = bw_acts.T
                bw_ims.append(np.array(bw_acts))
            full_img_data = bw_ims
        else:
            print('Unsupported Image Type')
        return(full_img_data)
            
    def process_images(self, full_img_data, patch_edge_size=None, normalize_im=False, 
                       normalize_patch=False, invert_colors=False):  
            if(normalize_im):
                print('normalizing full images...')
                full_img_data = full_img_data - np.mean(full_img_data,axis=(1,2),keepdims=True)
                full_img_data = full_img_data/np.std(full_img_data,axis=(1,2),keepdims=True)
            if(invert_colors):
                print('inverting colors...')
                full_img_data = full_img_data*(-1)
            if patch_edge_size is not None:
                print('sectioning into patches....')
                (num_img, num_px_rows, num_px_cols) = full_img_data.shape
                #crop to patch rows
                if(num_px_rows % patch_edge_size != 0):
                    nump = int(num_px_rows/patch_edge_size)
                    full_img_data = full_img_data[:,:nump*patch_edge_size,:]
                    (num_img, num_px_rows, num_px_cols) = full_img_data.shape
                #crop to patch cols
                if(num_px_cols % patch_edge_size != 0):
                    nump = int(num_px_cols/patch_edge_size)
                    full_img_data = full_img_data[:nump*patch_edge_size,:,:]
                    (num_img, num_px_rows, num_px_cols) = full_img_data.shape
                num_img_px = num_px_rows * num_px_cols
                #calc number of patches & calculate them
                self.num_patches = int(num_img_px / patch_edge_size**2)                
                data = np.asarray(np.split(full_img_data, num_px_cols/patch_edge_size,2)) # tile column-wise
                data = np.asarray(np.split(data, num_px_rows/patch_edge_size,2)) #tile row-wise
                data = np.transpose(np.reshape(np.transpose(data,(3,4,0,1,2)),(patch_edge_size,patch_edge_size,-1)),(2,0,1)) #stack tiles together
                if(normalize_patch):
                    print('normalizing patches...')
                    data = data - np.mean(data,axis=(1,2),keepdims=True)
                    data = data/np.std(data,axis=(1,2),keepdims=True)
            else:
                data = full_img_data
                self.num_patches = 0
            return data
        
        
#Load in images 
def loadimages(img_dir, psz):
    print("Loading Natural Image Database...")
    vhimgs = imageFile(
        img_dir = img_dir,
        normalize_im = True,
        normalize_patch = False,
        invert_colors = False,
        patch_edge_size=psz
        )
    print("Done Loading!")    
    np.random.shuffle(vhimgs.images)
    print("Done Shuffling!")
    return(vhimgs, psz)

#check for patchsize
def check_n_load_ims(img_dir,psz):
    try:
        vhimgs
    except NameError:
        vhimgs, loadedpatchsize = loadimages(img_dir, psz)

    if(psz != loadedpatchsize):
        vhimgs, loadedpatchsize = loadimages(img_dir, psz)

    print("Images Loaded.")

    #params of images
    imxlen = len(vhimgs.images[0,0,:])
    imylen = len(vhimgs.images[0,:,0])
    nimages = len(vhimgs.images[:,0,0])
    
    return(vhimgs, nimages)
