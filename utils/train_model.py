import tensorflow as tf
import numpy as np
from IPython.display import clear_output
import utils.plotutils as plu
import utils.conversions as cnv
import utils.movie_plotutils as mplu

import matplotlib.pyplot as plt
from time import clock
import os


        
#make session and train model
def train_movie_model(aec,m):
 
    with tf.device('/device:GPU:2'):
        config = tf.ConfigProto()
        #config.gpu_options.allow_growth = True #don't allocate the entire GPU's memory
        config.log_device_placement=True #tell me where devices are placed
        with tf.Session(graph = aec.graph, config = config) as sess:

            #initialize vars
            init = tf.global_variables_initializer()
            sess.run(init)

            #summary writer for tensorboard
            writer = tf.summary.FileWriter(aec.params['savefolder'],
                                           graph=tf.get_default_graph())

            #save evolution of system over training
            cost_evolution = []
            inweights_evolution = []
            outweights_evolution = []
            activation_evolution = []
            clips = []
            recons = []
            
            print('{} hidden neurons, noise_in at {}, noise_out at {}'.format(aec.params['nneurons'], aec.params['noise_x'], aec.params['noise_r']))

            print('Training {} iterations in {} epochs... '.format(aec.params['iterations'],
                                                                   aec.params['epochs']))
            for epoch in range(aec.params['epochs']):
                #start = clock()
                #print('Epoch {}: '.format(epoch+1))
                np.random.shuffle(m)
                #print('shuffled.')
                #print ('Time_shuffle:', (clock()-start))
                os.system('clear')
                print('Epoch {}/{} '.format(epoch, aec.params['epochs']), end='')
                for ii in range(aec.params['iterations']):
                    #reshape our images for feeding to dict
                    clip = m[ii*aec.params['batchsize']:(1+ii)*aec.params['batchsize'],:,:,:].astype(np.float32)

                    #setup params to send to dictionary
                    feeddict = {aec.x: clip}
                    
                    #run our session
                    sess.run(aec.train_step, feed_dict=feeddict)

                    #yin = sess.run(aec.yin, feed_dict=feeddict)
                    #print('yin_sum: ', np.sum(np.absolute(yin)))
                    #print('yin_mean: ', np.mean(yin))


                    #save evolution of params
                    objcost = sess.run([aec.cost], feed_dict=feeddict)
                    cost_evolution.append(objcost)

                    #save detailed parameters 10 times over the total evolution
                    if(((aec.params['iterations']*epoch)+ii) % (int((aec.params['iterations'] * aec.params['epochs'])/10))==0):
                        act, win, wout, clip, recon = sess.run([aec.mean_act,
                                                               aec.win, 
                                                               aec.wout, 
                                                               aec.x,
                                                               aec.xp], 
                                                               feed_dict=feeddict)
                        
#                         win = np.reshape(win,
#                                          (aec.params["framepatchsize"],
#                                           aec.params["pixelpatchsize"],
#                                           aec.params["pixelpatchsize"],
#                                           aec.params['nneurons']))
                                         
#                         wout = np.reshape(wout,
#                                          (aec.params['nneurons'],
#                                          aec.params["pixelpatchsize"],
#                                          aec.params["pixelpatchsize"],
#                                          aec.params["framepatchsize"]))
                                         
                        #save our weights, image, and reconstruction
                        inweights_evolution.append(win) #np.transpose(win, (0,1,2,4,3)))
                        outweights_evolution.append(wout)#np.transpose(wout, (0,1,2,4,3)))
                        activation_evolution.append(act)
                        clips.append(clip)#(clip_transformed[0][0])
                        recons.append(recon)#(recons_transformed[0][0])
                        
                        #plot example
                        clear_output()
                        mplu.plot_temporal_weights(wout.T)
                        plt.show()

                #end = clock()
                #print ('Time:', "%.1f" % (end-start),'sec')

            #summarize final params
            objcost, final_inweights, final_outweights, final_activations = sess.run([aec.cost, aec.win, aec.wout, aec.activation], feed_dict=feeddict)
            
#             final_inweights = np.reshape(final_inweights,
#                              (aec.params["framepatchsize"],
#                               aec.params["pixelpatchsize"],
#                               aec.params["pixelpatchsize"],
#                               aec.params['nneurons']))

#             final_outweights = np.reshape(final_outweights,
#                              (aec.params['nneurons'],
#                              aec.params["pixelpatchsize"],
#                              aec.params["pixelpatchsize"],
#                              aec.params["framepatchsize"]))
           
        print('Done calculating!')

        return(cost_evolution,
               activation_evolution,
               inweights_evolution,
               outweights_evolution,
               clips,
               recons,
               final_inweights,
               final_outweights,
               final_activations)