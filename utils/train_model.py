import tensorflow as tf
import numpy as np

def train_model(aec,vhimgs,tvhimgs):
    with tf.device("/gpu:0"):
        config = tf.ConfigProto()
        config.gpu_options.allow_growth = True #don't allocate the entire GPU's memory
        config.log_device_placement=True #tell me where devices are placed
        with tf.Session(graph = aec.graph, config=config) as sess:

            #initialize vars
            init = tf.global_variables_initializer()
            sess.run(init)

            #summary writer for tensorboard
            writer = tf.summary.FileWriter(aec.params['savefolder'],
                                           graph=tf.get_default_graph())

            #save evolution of system over training
            cost_evolution = []
            wmean_evolution = []

            inweights_evolution = []
            outweights_evolution = []
            inbias_evolution = []
            activation_evolution = []

            activations = []

            images = []
            recons = []
            print('neurons={}, noise_in={}, noise_out={}, lambda_w={}, lambda_act={}'
                  .format(aec.params['nneurons'],
                          aec.params['noise_x'],
                          aec.params['noise_r'],
                          aec.params['lambda_wgt'],
                          aec.params['lambda_act']))

            print('Training {} iterations in {} epochs... '.format(aec.params['iterations'],
                                                                   aec.params['epochs']))
            for epoch in range(aec.params['epochs']):
                #print('Epoch {}: '.format(epoch+1))
                np.random.shuffle(vhimgs)
                for ii in range(aec.params['iterations']):

                    #reshape our images for feeding to dict
                    image = np.reshape(vhimgs[ii*aec.params['batchsize']:(1+ii)*aec.params['batchsize'],:,:],
                                       (aec.params['batchsize'],
                                        aec.params['imxlen']*aec.params['imylen'])).astype(np.float32)

                    #setup params to send to dictionary
                    feeddict = {aec.x: image}

                    #run our session
                    sess.run(aec.train_step, feed_dict=feeddict)

                    #save evolution of params
                    objcost, inws, acts = sess.run([aec.recon_err, aec.win, aec.activation], feed_dict=feeddict)  #aec.cost
                    cost_evolution.append(objcost)
                    wmean_evolution.append(np.mean(np.abs(inws)))
                    activations.append(np.mean(acts,axis=0))

                    #save detailed parameters 10 times over the total evolution
                    if(ii%(int((aec.params['iterations']*aec.params['epochs'])/10))==0):
                        print(str(ii)+', ',end="")
                        #dump our params
                        win, wout, img, recon, inbias, activation = sess.run([aec.win, aec.wout, aec.x, aec.xp, aec.inbias, aec.activation], feed_dict=feeddict)
                        #save our weights, image, and reconstruction
                        inweights_evolution.append(win)
                        outweights_evolution.append(wout)
                        inbias_evolution.append(inbias)
                        activation_evolution.append(activation)
                        
                        #reshape images and append
                        imshape = [aec.params['batchsize'],
                                   aec.params['imxlen'],
                                   aec.params['imylen']]   
                        images.append(np.reshape(img, imshape))
                        recons.append(np.reshape(recon, imshape))
                        

            #summarizeparams
            inweights_evolution = np.array(inweights_evolution)
            outweights_evolution = np.array(outweights_evolution)
            activation_evolution = np.mean(activation_evolution,axis=1)
            
            print('Running Test Set...')
            timages = np.reshape(tvhimgs,(tvhimgs.shape[0],
                                        aec.params['imxlen']*aec.params['imylen'])).astype(np.float32)

            #setup params to send to dictionary & run
            feeddict = {aec.x: timages}
            summary, cost, test_patches, test_recons, test_win, test_wout, test_acts, weights_kernel_in = sess.run([aec.summary_op, aec.cost, aec.x, aec.xp, aec.win, aec.wout, aec.activation, aec.weights_kernel_in], feed_dict=feeddict)

            #find order based on activations
            order_test_acts = np.argsort(-np.mean(np.array(test_acts),axis=0))

            #reorder our data based on this ordering
            test_inweights_ordered = test_win[:,order_test_acts] #reorder based on activations
            test_outweights_ordered = test_wout.T[:,order_test_acts] #reorder based on activations
            test_acts_ordered = test_acts[:,order_test_acts] #reorder based on activations
            
            #reshape test image patches
            test_patches = np.reshape(test_patches, (test_patches.shape[0], aec.params['imxlen'], aec.params['imylen']))
            test_recons = np.reshape(test_recons, (test_patches.shape[0], aec.params['imxlen'], aec.params['imylen']))
            
            #reshape weight matrices
            #test_win = np.reshape(test_win.T, (aec.params['nneurons'], aec.params['imxlen'], aec.params['imylen']))
            #test_wout = np.reshape(test_wout, (aec.params['nneurons'], aec.params['imxlen'], aec.params['imylen']))
            
            #save summary
            writer.add_summary(summary,ii)
            writer.close()


            return(aec,
                   cost_evolution,
                   wmean_evolution,
                   inweights_evolution,
                   outweights_evolution,                   
                   activation_evolution,
                   inbias_evolution,
                   weights_kernel_in,
                   test_patches,
                   test_recons,
                   test_inweights_ordered,
                   test_outweights_ordered,
                   test_acts_ordered,
                   test_acts)