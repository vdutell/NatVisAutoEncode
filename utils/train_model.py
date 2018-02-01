import tensorflow as tf
import numpy as np
from IPython.display import clear_output
import utils.plotutils as plu
import utils.conversions as cnv

import matplotlib.pyplot as plt
from time import clock



def train_model(aec_model,vhimgs,tvhimgs):
    with tf.device("/gpu:0"):
        config = tf.ConfigProto()
        config.gpu_options.allow_growth = True #don't allocate the entire GPU's memory
        config.log_device_placement=True #tell me where devices are placed
        with tf.Session(graph = aec_model.graph, config=config) as sess:

            #initialize vars
            init = tf.global_variables_initializer()
            sess.run(init)

            #summary writer for tensorboard
            writer = tf.summary.FileWriter(aec_model.params['savefolder'],
                                           graph=tf.get_default_graph())

            #save evolution of system over training
            cost_evolution = []
            wmean_evolution = []

            inweights_evolution = []
            outweights_evolution = []
            inbias_evolution = []
            activations_evolution = []

            activations = []

            images = []
            recons = []
            print('neurons={}, noise_in={}, noise_out={}'
                  .format(aec_model.params['nneurons'],
                          aec_model.params['noise_x'],
                          aec_model.params['noise_r']))

            print('Training {} iterations in {} epochs... '.format(aec_model.params['iterations'],
                                                                   aec_model.params['epochs']))
            for epoch in range(aec_model.params['epochs']):
                #print('Epoch {}: '.format(epoch+1))
                np.random.shuffle(vhimgs)
                for ii in range(aec_model.params['iterations']):

                    #reshape our images for feeding to dict
                    image = np.reshape(vhimgs[ii*aec_model.params['batchsize']:(1+ii)*aec_model.params['batchsize'],:,:],
                                       (aec_model.params['batchsize'],
                                        aec_model.params['imxlen']*aec_model.params['imylen'])).astype(np.float32)

                    #setup params to send to dictionary
                    feeddict = {aec_model.x: image}

                    #run our session
                    sess.run(aec_model.train_step, feed_dict=feeddict)

                    #save evolution of params
                    objcost, inws, acts = sess.run([aec_model.cost, aec_model.win, aec_model.mean_act], feed_dict=feeddict)  #aec_model.cost
                    cost_evolution.append(objcost)
                    wmean_evolution.append(np.mean(np.abs(inws)))
                    activations.append(np.mean(acts,axis=0))

                    #save detailed parameters 10 times over the total evolution
                    if(ii%(int((aec_model.params['iterations']*aec_model.params['epochs'])/10))==0):
                        print(str(ii)+', ',end="")
                        #dump our params
                        win, wout, img, recon, inbias, activation = sess.run([aec_model.win, aec_model.wout, aec_model.x, aec_model.xp, aec_model.inbias, aec_model.activation], feed_dict=feeddict)
                        #save our weights, image, and reconstruction
                        inweights_evolution.append(win)
                        outweights_evolution.append(wout)
                        inbias_evolution.append(inbias)
                        activation_evolution.append(activation)
                        
                        #reshape images and append
                        imshape = [aec_model.params['batchsize'],
                                   aec_model.params['imxlen'],
                                   aec_model.params['imylen']]   
                        images.append(np.reshape(img, imshape))
                        recons.append(np.reshape(recon, imshape))
                        

                        #update on progess and plot our weights evolving
                        clear_output()
                        print(str(ii+aec_model.params['iterations']*epoch) +'/' + str(aec_model.params['iterations']*aec_model.params['epochs']), end="")
                        plu.plot_tiled_rfs(win.reshape(
                                    aec_model.params['imxlen'],
                                    aec_model.params['imylen'],
                                    aec_model.params['nneurons']).T, normalize=False)
                        plt.show()
                        
                    #show progress star
                    #print('*',end='')   
                        
            #summarizeparams
            inweights_evolution = np.array(inweights_evolution)
            outweights_evolution = np.array(outweights_evolution)
            activation_evolution = np.mean(activation_evolution,axis=1)
            
            print('Running Test Set...')
            timages = np.reshape(tvhimgs,
                                 tvhimgs.shape[0],
                                 aec_model.params['imxlen']*aec_model.params['imylen']).astype(np.float32)

            #setup params to send to dictionary & run
            feeddict = {aec_model.x: timages}
            summary, test_recon_err, test_patches, test_recons, test_win, test_wout, test_acts, weights_kernel_in = sess.run([aec_model.summary_op, aec_model.recon_err, aec_model.x, aec_model.xp, aec_model.win, aec_model.wout, aec_model.activation, aec_model.weights_kernel_in], feed_dict=feeddict)

            #find order based on activations
            order_test_acts = np.argsort(-np.mean(np.array(test_acts),axis=0))

            #reorder our data based on this ordering
            weights_kernel_in_ordered = weights_kernel_in[:,order_test_acts] #reorder based on activations
            test_inweights_ordered = test_win[:,order_test_acts] #reorder based on activations
            test_outweights_ordered = test_wout.T[:,order_test_acts] #reorder based on activations
            test_acts_ordered = test_acts[:,order_test_acts] #reorder based on activations
            
            #reshape test image patches
            test_patches = np.reshape(test_patches, (test_patches.shape[0], aec_model.params['imxlen'], aec_model.params['imylen']))
            test_recons = np.reshape(test_recons, (test_patches.shape[0], aec_model.params['imxlen'], aec_model.params['imylen']))
            
            #reshape weight matrices
            #test_win = np.reshape(test_win.T, (aec_model.params['nneurons'], aec_model.params['imxlen'], aec_model.params['imylen']))
            #test_wout = np.reshape(test_wout, (aec_model.params['nneurons'], aec_model.params['imxlen'], aec_model.params['imylen']))
            
            #save summary
            writer.add_summary(summary,ii)
            writer.close()

            print('Done!')
            return(aec_model,
                   cost_evolution,
                   wmean_evolution,
                   inweights_evolution,
                   outweights_evolution,                   
                   activation_evolution,
                   inbias_evolution,
                   weights_kernel_in_ordered,
                   test_recon_err,
                   test_patches,
                   test_recons,
                   test_inweights_ordered,
                   test_outweights_ordered,
                   test_acts_ordered,
                   test_acts)
        
        
        
#make session and train model
def train_movie_model(aec,m):
 
    with tf.device("/gpu:1"):
        config = tf.ConfigProto()
        config.gpu_options.allow_growth = True #don't allocate the entire GPU's memory
        config.log_device_placement=True #tell me where devices are placed
        with tf.Session(graph = aec.graph, config = config) as sess:

#             session_conf = tf.ConfigProto()
#             session_conf.gpu_options.allow_growth = True

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
            print('{} hidden neurons, noise_in at {}, noise_out at {}'.format(aec.params['nneurons'],
                                                                                            aec.params['noise_x'],
                                                                                            aec.params['noise_r']))

            print('Training {} iterations in {} epochs... '.format(aec.params['iterations'],
                                                                   aec.params['epochs']))
            for epoch in range(aec.params['epochs']):
                start = clock()
                #print('Epoch {}: '.format(epoch+1))
                np.random.shuffle(m)
                #print('shuffled.')
                #print ('Time_shuffle:', (clock()-start))
                print('Epoch {}'.format(epoch), end='')
                for ii in range(aec.params['iterations']):
                    print('*',end='')

                    #reshape our images for feeding to dict
                    clip = m[ii*aec.params['batchsize']:(1+ii)*aec.params['batchsize'],:,:,:].astype(np.float32)
                    #clip = np.reshape(m[ii*aec.params['batchsize']:(1+ii)*aec.params['batchsize'],:,:,:], 
                    #                  (aec.params['batchsize'],
                    #                   aec.params["framepatchsize"],
                    #                   aec.params["pixelpatchsize"],
                    #                   aec.params['pixelpatchsize'], 
                    #                   1)).astype(np.float32)

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
                    if(((aec.params['iterations']*epoch)+ii)%
                       (int((aec.params['iterations']*aec.params['epochs'])/10))==0):
                    #if((aec.params['iterations']*epoch)%10 ==0):
                        #dump our params
                        win, wout, act, clip, recon = sess.run([aec.win[:,:,:,0,:], 
                                                           aec.wout[:,:,:,0,:], 
                                                           aec.mean_act,
                                                           aec.x[:,:,:,:,0], 
                                                           aec.xp[:,:,:,:,0]], 
                                                           feed_dict=feeddict)
                        #save our weights, image, and reconstruction
                        inweights_evolution.append(win)#np.transpose(win, (0,1,2,4,3)))
                        outweights_evolution.append(win)#np.transpose(wout, (0,1,2,4,3)))
                        activation_evolution.append(act)
                        clips.append(clip)#(clip_transformed[0][0])
                        recons.append(recon)#(recons_transformed[0][0])

                end = clock()
                print ('Time:', "%.1f" % (end-start),'sec')

            #summarize final params
            objcost, final_inweights, final_outweights, final_activations = sess.run([aec.cost, aec.win[:,:,:,0,:],
                                                                                       aec.wout[:,:,:,0,:], 
                                                                                       aec.activation], 
                                                                                       feed_dict=feeddict)
            #cost_evolution.append(objcost)
           
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