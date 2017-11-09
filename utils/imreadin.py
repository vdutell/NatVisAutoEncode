import os
import numpy as np
import h5py
import glob
from scipy import io

class imageFile:
    def __init__(self,
                 imset,
                 patch_edge_size=None,
                 normalize_im=False,
                 patch_multiplier = 1,
                 normalize_patch=False,
                 invert_colors=False,
                 rand_state=np.random.RandomState()):
        
        # readin images
        self.images = self.extract_images(imset)
        # process images
        self.images = self.process_images(self.images, patch_edge_size, normalize_im, 
                                          patch_multiplier, normalize_patch, invert_colors)

    def extract_images(self, imset):
        #load in our images
        if(imset=='vh_lognorm'):
            self.image_files = '/home/vasha/datasets/vanHaterenNaturalImages/VanHaterenNaturalImagesCurated.h5'
            with h5py.File(self.image_files, "r") as f:
                full_img_data = np.array(f['van_hateren_good'], dtype=np.float32) 
        elif(imset=='kyoto'):
            self.image_files = '/home/vasha/datasets/eizaburo-doi-kyoto_natim-c2015ff/*.mat'
            bw_ims = []
            for file in glob.glob(self.image_files,recursive=True):
                mat = io.loadmat(file)
                #short medium and long activations
                sml_acts = np.array([mat['OS'],mat['OM'],mat['OL']])
                #mean over three for luminance
                bw_acts = np.mean(sml_acts,axis=0)
                # transpose if we need to ***I think this is allowed***
                if(np.shape(bw_acts)[0] > np.shape(bw_acts)[1]):
                    bw_acts = bw_acts.T
                bw_ims.append(np.array(bw_acts))
            full_img_data = np.array(bw_ims)
        elif(imset=='vh_corr'):
            imc_dir = '/home/vasha/datasets/vanHaterenNaturalImages/pirsquared/vanhateren_imc/*.imc'
            imdir = imc_dir #using optics corrected images

            dim = [1024,1536]
            full_img_data = []
            for file in glob.glob(imdir,recursive=True)[:100]:
                dtype = np.dtype ('uint16').newbyteorder('>')
                a = np.fromfile(file, dtype).reshape((dim))
                full_img_data.append(np.array(a))
            #normalize each image to max 1
            full_img_data = full_img_data/(np.amax(full_img_data,axis=(1,2))[:,None,None])
            #for j, im in enumerate(full_img_data):
            #    full_img_data[j] = im/np.amax(im,axis=(0,1))
            full_img_data = np.array(full_img_data)
            
        elif(imset=='vh_ncorr'):
            iml_dir = '/home/vasha/datasets/vanHaterenNaturalImages/pirsquared/vanhateren_iml/*.iml'
            imdir = iml_dir #using fully raw images
            dim = [1024,1536]
            ims = []
            for file in glob.glob(imdir,recursive=True)[:100]:
                dtype = np.dtype ('uint16').newbyteorder('>')
                a = np.fromfile(file, dtype).reshape((dim))
                ims.append(np.array(a))
            #normalize each image to max 1
            ims = ims/(np.amax(ims,axis=(1,2))[:,None,None])
            full_img_data = np.array(ims)
            
        else:
            print('%s is an unsupported Image Type!!!'.format(imset))
        return(full_img_data)

    def patch_maker(self, full_img_data, patch_edge_size, offset):
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
        return(data)
            
    def process_images(self, full_img_data, patch_edge_size=None, 
                       normalize_im=False, patch_multiplier = 1,
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
                data = []
                if(patch_multiplier>1):
                    print('multipying patches by {}...'.format(patch_multiplier))
                for i in range(patch_multiplier):
                    offset_px = patch_edge_size/patch_multiplier # size in pixels of each offset
                    data.append(np.array(self.patch_maker(full_img_data, patch_edge_size,offset=offset_px*i)))
                data = np.array(data)
                data = np.reshape(data,(-1,np.shape(data)[2],np.shape(data)[3]))
                print('now we have {} patches'.format(np.shape(data)[0]))
                self.num_patches = np.shape(data)[0]
            else:
                data = full_img_data
                self.num_patches = 1
            if(normalize_patch):
                print('normalizing patches...')
                data = data - np.mean(data,axis=(1,2),keepdims=True)
                data = data/np.std(data,axis=(1,2),keepdims=True)
            return data
        
        
#Load in images 
def loadimages(imset, psz, pm, norm=True):
    print("Loading Natural Image Database...")
    vhimgs = imageFile(
        imset = imset,
        normalize_im = norm,
        patch_multiplier = pm,
        normalize_patch = False,
        invert_colors = False,
        patch_edge_size=psz
        )
    print("Done Loading!")    
    np.random.shuffle(vhimgs.images)
    print("Done Shuffling!")
    return(vhimgs.images, psz)

#check for patchsize
def check_n_load_ims(imset, psz, pm):
    vhimgs, loadedpatchsize = loadimages(imset, psz, pm)

    print("Images Ready.")

    #params of images
    imxlen = len(vhimgs[0,0,:])
    imylen = len(vhimgs[0,:,0])
    nimages = len(vhimgs[:,0,0])
    
    return(vhimgs, nimages)
