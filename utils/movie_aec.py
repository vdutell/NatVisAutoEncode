import os
import tensorflow as tf
import numpy as np
import shutil
    
class movie_aec_model(object):
    
    def __init__(self, params):
        params = self.add_params(params)
        self.params = params
        self.make_dirs()
        self.graph = self.make_graph()
    
    def add_params(self, params):  
        params['compression'] = params['pixelpatchsize']*params['pixelpatchsize']/params['nneurons']
        params['savefolder'] = str('./output/movie_output/actfun_'+ params['nonlinearity']+
                                   '_hiddenneurons_'+ str(params['nneurons'])+
                                   '_noisein_'+ str(params['noise_x'])+
                                   '_noiseout_'+ str(params['noise_r'])+'/')
        return(params)
        
    def make_dirs(self):
        if not os.path.exists(self.params['savefolder']):
            os.makedirs(self.params['savefolder'])
        
    def make_graph(self):
    
        print('Compressing by',self.params['compression'],
              'for a total of',self.params['nneurons'],'neurons')

        #setup our graph
        #tf.reset_default_graph()
        mygraph = tf.Graph()
        with mygraph.as_default():

            #input images
            with tf.name_scope('input'):
                    self.x = tf.placeholder(tf.float32, shape=[self.params["batchsize"], 
                                                           self.params["framepatchsize"],
                                                           self.params["pixelpatchsize"],
                                                           self.params["pixelpatchsize"]])
                    #self.xt = tf.transpose(self.x,perm=(0,3,2,1))
#                     self.xvec = tf.reshape(self.x,[self.params["batchsize"], 
#                                                    params['clipvec_len']]
                   
            #activation function type
            with tf.name_scope('activation_function'):
                self.act_fun = self.params['nonlinearity']

            #noises
            with tf.name_scope('noises'):
                self.noisexsigma = self.params['noise_x']
                self.noisersigma = self.params['noise_r']

            #function to add noise
            with tf.name_scope("add_noise"):
                def add_noise(input_layer, std):
                    noise = tf.random_normal(shape=tf.shape(input_layer),
                                             mean=0.0,
                                             stddev=std,
                                             dtype=tf.float32) 
                    return tf.add(input_layer,
                                  noise)

            #weights
            with tf.variable_scope("weights"):
#                 weights_kernel = tf.random_normal([self.params['frames_per_channel'], 
#                                                    self.params['pixelpatchsize'], 
#                                                    self.params['pixelpatchsize'],
#                                                    1,
#                                                    self.params['nneurons']],
#                                                    dtype=tf.float32,stddev=0.1)
#                weights_kernel = tf.random_normal([self.params['frames_per_channel'],
#                                                   self.params['pixelpatchsize'], 
#                                                   self.params['pixelpatchsize'],
#                                                   self.params['nneurons']],
#                                                   dtype=tf.float32)
                weights_kernel = tf.random_uniform([self.params['frames_per_channel'],
                                                   self.params['pixelpatchsize'], 
                                                   self.params['pixelpatchsize'],
                                                   self.params['nneurons']],
                                                   dtype=tf.float32,
                                                   minval=-1)
                                           
#                 weights_kernel = tf.random_normal([params['clipvec_len'],
#                                                    self.params['nneurons']]

                self.win = tf.get_variable(name='weights_in',
                                           initializer = weights_kernel)
    
                self.wout = tf.get_variable(name='weights_out',
                                            initializer=tf.transpose(weights_kernel))
            
                #self.wout@tf.transpose(self.wout)

                wnormalizer = tf.norm(tf.reshape(self.win,
                                                (-1, self.params['nneurons'])),
                                      ord='euclidean',
                                      axis=0)
                
                wnormalizer = tf.reshape(wnormalizer,
                                         (1,1,1,-1))
                self.win = self.win * (1./wnormalizer)
            
                #self.wout = tf.transpose(self.win)
                #self.wout = tf.get_variable('weights_out', initializer=tf.transpose(weights_kernel))
                
#                 self.wout = tf.get_variable('weights_out',[self.params['nneurons'],
#                                                            self.params['pixelpatchsize'], 
#                                                            self.params['pixelpatchsize']],
#                                                            dtype=tf.float32)
                                            
                #self.wout = tf.get_variable('weights_out',initializer=weights_kernel)
                #self.wout = tf.get_variable('weights_out',initializer=tf.transpose(weights_kernel))

            #bias
            with tf.variable_scope("bias"):
                self.bias = tf.zeros([self.params['nneurons']],
                                     dtype=tf.float32) #tf.Variable(tf.random_normal([self.params['nneurons']],dtype=tf.float32))

            #learning_rate
            with tf.name_scope('learning_rate'):
                self.learning_rate = self.params['learning_rate']

            #nonlienarities
            with tf.name_scope("nonlienarities"):
                #define nonlinearities
                def tanh_fun(arg):
                    return tf.nn.tanh(arg) 
                def sigmoid_fun(arg):
                    return tf.nn.sigmoid(arg) 
                def relu_fun(arg):
                    return tf.nn.relu(arg) 
                def no_fun(arg):
                    return arg

            #encoding part of model
            with tf.name_scope("encoding"):
                #calculate input
                
                noised_input = add_noise(self.x,self.params['noise_x'])
                #expand dims for 1 channel: Need to change this for color channel
                #linearin = tf.nn.conv3d(noised_input, self.win, strides= [1,
                #                                                          1,
                #                                                          self.params['pixelpatchsize'], 
                #                                                          self.params['pixelpatchsize'],
                #                                                          1], 
                #                       padding='SAME') #Convolution over time, and multiply by weight
                
                linearin = tf.einsum('ijkl,mijk->ml', self.win, noised_input) #[500,5,12,12], [5,12,12,144] -> [500,144])
                #linearin = self.win @ noised_input
                
                #linearin = tf.add(linearin,self.bias)
                                        
                self.activation = tf.case({tf.equal(self.act_fun,'tanh'): (lambda: tanh_fun(linearin)),
                               tf.equal(self.act_fun,'sigmoid'): (lambda: sigmoid_fun(linearin)),
                               tf.equal(self.act_fun,'relu'): (lambda: relu_fun(linearin))},
                               default=(lambda: no_fun(linearin)),
                               exclusive=True)
                #self.yin = add_noise(self.activation,self.params['noise_r'])
                self.yin = self.activation  
            
            #output part of model
                                        
            with tf.name_scope("decoding"):
                #calculate output (reconstruction)
                
#                 self.xp = tf.nn.conv3d_transpose(self.yin, self.wout, 
#                                                  output_shape = (self.params["batchsize"],
#                                                                  self.params["pixelpatchsize"],
#                                                                  self.params["pixelpatchsize"], 
#                                                                  1,1), 
#                                                  strides=[1,
#                                                           1,
#                                                           self.params['pixelpatchsize'], 
#                                                           self.params['pixelpatchsize'],
#                                                           1], 
#                                                  padding='SAME') #Deconvolution 
                 
                
#                 self.xpvec = self.yin @ self.woutvec
               
#                 self.xp = tf.reshape(self.xpvec,[self.params["batchsize"],
#                                                  self.params["pixelpatchsize"],
#                                                  self.params["pixelpatchsize"],
                                                 
                self.xp = tf.einsum('ij,jklm->imlk', self.yin, self.wout)   # [500,144],[144,5,12,12] -> [500,5,12,12]   
                #self.xp = tf.matmul(self.yin,self.wout) #add noise to inner layer, and multiply by weight transpose
                #self.xp = tf.case({tf.equal(self.act_fun,'tanh'): (lambda: tanh_fun(linearout)),
                #                    tf.equal(self.act_fun,'sigmoid'): (lambda: sigmoid_fun(linearout)),
                #                    tf.equal(self.act_fun,'relu'): (lambda: relu_fun(linearout))},
                #                    default=(lambda: no_fun(linearout)),
                #                    exclusive=True, name='output_nonlienarity')
             
            #lambda activation
            with tf.name_scope('lambda_activation'):
                self.mean_act = tf.reduce_sum(tf.reduce_mean(self.activation,axis=0),axis=0)
                desired_spikes_per_neuron = self.params["framepatchsize"]/self.params["frames_per_channel"]
                self.lambda_act = tf.abs(desired_spikes_per_neuron-self.mean_act) #no cost if avg activation is as expectd 
            
            
            #self.xp @ tf.transpose(self.x)
            #self.x@tf.transpose(self.xp)
            
            #vectorize before calculating norm
            with tf.name_scope('recon_error'):
                #self.recon_err = (tf.reshape(self.x,[self.params["batchsize"],-1]) - 
                #                  tf.reshape(self.xp,[self.params["batchsize"],-1]))
                #(self.x - self.xp) @ tf.transpose(self.xp)
                self.recon_err = tf.reshape((self.x - self.xp),[self.params["batchsize"],-1])
                self.recon_err = tf.norm(self.recon_err, ord='euclidean', axis=1)
                #self.recon_err = tf.reduce_mean(tf.norm(self.x-self.xp, ord='euclidean', axis=(2,3)))

            #calculate cost
            with tf.name_scope("cost_function"):
                #self.lambda_act = tf.reduce_mean(self.activation+1e-5, axis=0)
                self.cost = self.recon_err + tf.reduce_mean(self.lambda_act)
 
            #train our model
            with tf.name_scope("training_step"):
                self.train_step = tf.train.AdamOptimizer(self.learning_rate).minimize(self.cost)

            # create a summary for our cost, im, reconstruction, & weights
            with tf.name_scope('cost_viz'):
                tf.summary.scalar("cost", self.cost)

            #with tf.name_scope('image_viz'):    
            #    x_t = tf.reshape(self.x,(self.params['batchsize'], 0, self.params['imxlen'],self.params['imylen'],1))
            #    tf.summary.image("image", x_t, max_outputs=self.params["batchsize"])

            #with tf.name_scope('recon_viz'):
            #    xp_t = tf.reshape(self.xp,(self.params['batchsize'], 0,self.params['imxlen'],self.params['imylen'],1))
            #    print (xp_t)
            #    tf.summary.image("recon", xp_t,max_outputs=self.params["batchsize"])
#CHANGE
            #with tf.name_scope('inweights_viz'):    
            #    #inwin_t = tf.reshape(tf.transpose(self.win),
            #    #inwin_t = tf.reshape(tf.transpose(self.win, perm=[3,0,1,2]), (self.params['nneurons'], self.params['imxlen'], self.params['imylen'],1))
            #    inwin_t = tf.transpose(self.win, perm=[0,4,1,2,3])
            #    tf.summary.image("inweights", inwin_t, max_outputs=self.params['nneurons'])
                
            #with tf.name_scope('outweights_viz'):    
            #    #outwin_t = tf.reshape(tf.transpose(self.wout[0], perm=[3,0,1,2]), (self.params['nneurons'], self.params['imxlen'], self.params['imylen'],1))
            #    outwin_t = tf.transpose(self.wout, perm=[0,4,1,2,3])
            #    tf.summary.image("outweights", outwin_t, max_outputs=self.params['nneurons'])
                      
            #with tf.name_scope('activnonlin_viz'):
            #    activation = tf.transpose(self.yin, perm=[0,4,1,2,3])
            #    activation = tf.reshape(activation, (self.params["batchsize"], self.params["time_patchsize"], -1))  #reshape nonlinear-vector [batchsize, nneurons, time] 
            #    activ_help_temp = activation[0,0] # temporarily take only one nneuron over time
            #    tf.summary.scalar("activnonlin", activ_help_temp)     
                
                
            # merge all summaries into a single "operation" which we can execute in a session 
            #self.summary_op = tf.summary.merge_all()

        return(mygraph)
    