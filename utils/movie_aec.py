import os
import tensorflow as tf
import numpy as np
import shutil

class aec_model(object):
    
    def __init__(self, params):
        params = self.add_params(params)
        self.params = params
        self.make_dirs()
        self.graph = self.make_graph()

    def add_params(self, params):  
        params['compression'] = params['imxlen'] * params['imylen'] / params['nneurons']
        params['savefolder'] = str('./output/image_output/' + 
                                   str(params['ims']) +
                                   '_' + str(params['nimgs']) +
                                   '_nonlin1_' + str(params['nonlin1'])+ 
                                   '_nonlin2_' + str(params['nonlin2'])+
                                   '_neurons_'+ str(params['nneurons'])+
                                   '_nin_'+ str(params['noise_x'])+
                                   '_nout_'+ str(params['noise_r'])+
                                   '_bsze_'+ str(params['batchsize'])+
                                   '_epochs_'+ str(params['epochs'])+
                                   '_lrate_'+ str(params['learning_rate'])+
                                   '_invertcolors_' + str(params['colorinvert']) + '/')

        return(params)
        
    def make_dirs(self):
        if os.path.exists(self.params['savefolder']):
            shutil.rmtree(self.params['savefolder'])
        os.makedirs(self.params['savefolder'])
        os.makedirs(self.params['savefolder']+'param_evolution/')
        
    def make_graph(self):
    
        print('Compressing by',self.params['compression'],'for a total of',self.params['nneurons'],'neurons')

        #setup our graph
        #tf.reset_default_graph()
        mygraph = tf.Graph()
        with mygraph.as_default():
            
            #input images
            with tf.name_scope('input'):
                self.x = tf.placeholder(tf.float32, shape=[None, 
                                                           self.params["imxlen"]*self.params["imylen"]])
            #now can define batch size
            #self.nbatches = tf.constant([self.params['batchsize'],1],dtype='int32')
            batch_size = tf.shape(self.x)[0]
         
            #activation function type
            with tf.name_scope('nonliearities'):
                self.nonlin1 = self.params['nonlin1']
                self.nonlin2  = self.params['nonlin2']

            #noises
            with tf.name_scope('noises'):
                self.noisexsigma = self.params['noise_x']
                self.noisersigma = self.params['noise_r']

            #function to add noise
            with tf.name_scope("add_noise"):
                def add_noise(input_layer, std):
                    noise = tf.random_normal(shape=tf.shape(input_layer), mean=0.0, stddev=std, dtype=tf.float32) 
                    return tf.add(input_layer,noise)
                             
            #weights
            with tf.variable_scope("weights"):

                #per Salimans et al 2016 - Weight Normalization: A Simple Reparameterization to Accelerate Training of Deep Neural Networks parameterize weights as w = g/||v||*v. Now ||w|| = g and we can set g to 1 to enfoce this constraint, and learn the direction of the weights v, while maintaining magnitude of norm 1.

                self.weights_kernel_in = tf.random_uniform([self.params['imxlen']*self.params['imylen'],
                                                      self.params['nneurons']],
                                                     dtype=tf.float32,minval=-1,maxval=1)
                
                self.win = tf.get_variable('win', initializer = self.weights_kernel_in)
                
                # normalize weights
                self.win = self.win / tf.norm(self.win,  ord=2, axis=0)
                
                #self.lambda_weight = tf.norm(self.win,  ord=2, axis=0)-1 #SHAPE IS 100
                #self.mean_act = tf.reduce_mean(self.activation,axis=0)
                #self.lambda_act = tf.norm(self.mean_act-1, ord=1)
            

                self.wout = tf.get_variable('wout', initializer=tf.transpose(self.win))
                
                
            #bias
            with tf.variable_scope("in_bias"):
                offset = 0 #to keep values positive
                self.inbias = tf.Variable(tf.random_normal([self.params['nneurons']],
                                                         dtype=tf.float32,
                                                         stddev=0.1)+offset)
            with tf.variable_scope("out_bias"):
                offset = 0
                if(self.params['nonlin2']in['sigmoid','relu','tanh']):
                    self.outbias = tf.Variable(tf.random_normal([self.params['imxlen']*self.params['imylen']],
                                                                dtype=tf.float32, stddev=0.1)+offset)
                else:
                    self.outbias = tf.zeros([self.params['imxlen']*self.params['imylen']])

             #learning_rate
            with tf.name_scope('learning_rate'):
                self.learning_rate = self.params['learning_rate']

            #nonlienarities
            with tf.name_scope("nonlienarities"):
                #define nonlinearities
                def tanh_fun(bias,arg):
                    return tf.nn.tanh(tf.add(arg,bias)) 
                def sigmoid_fun(bias,arg):
                    return tf.nn.sigmoid(tf.add(arg,bias)) 
                def relu_fun(bias,arg):
                    return tf.nn.relu(tf.add(arg,bias)) 
                def no_fun(bias,arg):
                    return arg

            #encoding part of model
            with tf.name_scope("encoding"):
                
                #self.win = tf.nn.l2_normalize(self.win, dim=0)
                linearin = tf.matmul(add_noise(self.x,self.params['noise_x']),self.win) #add noise to input, and multiply by weights
                self.activation = tf.case({tf.equal(self.nonlin1,'tanh'): (lambda: tanh_fun(self.inbias,linearin)),
                               tf.equal(self.nonlin1,'sigmoid'): (lambda: sigmoid_fun(self.inbias,linearin)),
                               tf.equal(self.nonlin1,'relu'): (lambda: relu_fun(self.inbias,linearin))},
                               default=(lambda: no_fun(self.inbias,linearin)),
                               exclusive=True)
                self.yin_noised = add_noise(self.activation,self.params['noise_r'])
                
            #output part of model
            with tf.name_scope("decoding"):
                #calculate output (reconstruction)
                linearout = tf.matmul(self.yin_noised,self.wout) #add noise to inner layer, and multiply by weight  transpose
                self.xp = tf.case({tf.equal(self.nonlin2,'tanh'): (lambda: tanh_fun(self.outbias,linearout)),
                                   tf.equal(self.nonlin2,'sigmoid'): (lambda: sigmoid_fun(self.outbias,linearout)),
                                   tf.equal(self.nonlin2,'relu'): (lambda: relu_fun(self.outbias,linearout))},
                                   default=(lambda: no_fun(self.outbias,linearout)),
                                   exclusive=True, name='output_nonlienarity')
            
            #how well are we reconstructing?
            with tf.name_scope("reconstruction"):
                self.recon_err = tf.reduce_mean(tf.norm(self.x-self.xp, ord=2, axis=1))
                
            #lambda activation
            with tf.name_scope('lambda_activation'):
                #self.lambda_act = tf.Variable(tf.ones([self.params['nneurons']]),dtype=tf.float32)
                self.mean_act = tf.reduce_mean(self.activation,axis=0)

                self.lambda_act = tf.reduce_mean(tf.abs(1-self.mean_act)) #no cost if avg activation is 1       
                    
            #calculate cost
            with tf.name_scope("cost_function"):
                                  
                #self.cost = self.recon_err + tf.reduce_mean(self.lambda_act * self.mean_act)
                self.cost = self.recon_err + 0.01 * self.lambda_act #+ tf.reduce_sum(self.lambda_weight)
                #100*tf.reduce_mean(self.lambda_act * self.activation, axis=1)
            #train our model
            with tf.name_scope("training_step"):
                self.train_step = tf.train.AdamOptimizer(self.learning_rate).minimize(self.cost)
                #self.train_step = tf.train.GradientDescentOptimizer(self.learning_rate).minimize(self.cost)
                
            # create a summary for our cost, im, reconstruction, & weights
            with tf.name_scope('cost_viz'):
                tf.summary.scalar("cost", tf.reduce_mean(self.cost))

            with tf.name_scope('image_viz'):    
                x_t = tf.reshape(self.x,(batch_size,self.params['imxlen'],self.params['imylen'],1))
                tf.summary.image("image", x_t, max_outputs=self.params['batchsize'])

            with tf.name_scope('recon_viz'):
                xp_t = tf.reshape(self.xp,(batch_size,self.params['imxlen'],self.params['imylen'],1))
                tf.summary.image("recon", xp_t,max_outputs=self.params['batchsize'])

            with tf.name_scope('inweights_viz'):    
                inwin_t = tf.reshape(tf.transpose(self.win),
                                   (self.params['nneurons'],
                                    self.params['imxlen'],
                                    self.params['imylen'],1))
                tf.summary.image("inweights", inwin_t, max_outputs=self.params['nneurons'])
                
            with tf.name_scope('outweights_viz'):    
                outwin_t = tf.reshape(self.wout,
                                   (self.params['nneurons'],
                                    self.params['imxlen'],
                                    self.params['imylen'],1))
                tf.summary.image("outweights", outwin_t, max_outputs=self.params['nneurons'])

            # merge all summaries into a single "operation" which we can execute in a session 
            self.summary_op = tf.summary.merge_all()

        return(mygraph)

    
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
                                                           self.params["pixelpatchsize"],
                                                           1])

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
                    noise = tf.random_normal(shape=tf.shape(input_layer), mean=0.0, stddev=std, dtype=tf.float32) 
                    return tf.add(input_layer,noise)

            #weights
            with tf.variable_scope("weights"):
                weights_kernel = tf.random_normal([self.params['frames_per_channel'], 
                                                   self.params['pixelpatchsize'], 
                                                   self.params['pixelpatchsize'],
                                                   1,
                                                   self.params['nneurons']],
                                                   dtype=tf.float32,stddev=0.1)
                
                self.win = tf.get_variable('weights_in',initializer=weights_kernel)
                self.win = self.win * tf.expand_dims(1 / tf.reduce_mean(tf.norm(self.win,  ord='euclidean', axis=(1,2)),axis=0),0)

                self.wout = tf.get_variable('weights_out',initializer=weights_kernel)
                #self.wout = tf.get_variable('weights_out',initializer=tf.transpose(weights_kernel))

            #bias
            with tf.variable_scope("bias"):
                self.bias = tf.Variable(tf.random_normal([self.params['nneurons']],dtype=tf.float32,stddev=0.1))

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
                linearin = tf.nn.conv3d(noised_input, self.win, strides= [1,1,self.params['pixelpatchsize'], 
                                                                                self.params['pixelpatchsize'],1], 
                                               padding='SAME') #Convolution over time, and multiply by weight
                
                linearin = tf.add(linearin,self.bias)
                                        
                self.activation = tf.case({tf.equal(self.act_fun,'tanh'): (lambda: tanh_fun(linearin)),
                               tf.equal(self.act_fun,'sigmoid'): (lambda: sigmoid_fun(linearin)),
                               tf.equal(self.act_fun,'relu'): (lambda: relu_fun(linearin))},
                               default=(lambda: no_fun(linearin)),
                               exclusive=True)
                self.yin = add_noise(self.activation,self.params['noise_r'])
               
            #output part of model
                                        
            with tf.name_scope("decoding"):
                #calculate output (reconstruction)
                
                self.xp = tf.nn.conv3d_transpose(self.yin, self.wout, 
                                                 output_shape = (self.params["batchsize"],
                                                                 self.params["framepatchsize"], 
                                                                 self.params["pixelpatchsize"],
                                                                 self.params["pixelpatchsize"], 1), 
                                                 strides=[1,1,self.params['pixelpatchsize'], 
                                                          self.params['pixelpatchsize'],1], 
                                                 padding='SAME') #Deconvolution 
                                        
                #self.xp = tf.matmul(self.yin,self.wout) #add noise to inner layer, and multiply by weight transpose
                #self.xp = tf.case({tf.equal(self.act_fun,'tanh'): (lambda: tanh_fun(linearout)),
                #                    tf.equal(self.act_fun,'sigmoid'): (lambda: sigmoid_fun(linearout)),
                #                    tf.equal(self.act_fun,'relu'): (lambda: relu_fun(linearout))},
                #                    default=(lambda: no_fun(linearout)),
                #                    exclusive=True, name='output_nonlienarity')
             
            #lambda activation
            with tf.name_scope('lambda_activation'):
                self.mean_act = tf.reduce_sum(tf.reduce_mean(self.activation,axis=0),axis=0)[0,0,:] #mean over batches, sum over frames
                desired_spikes_per_neuron = self.params["framepatchsize"]/self.params["frames_per_channel"]
                self.lambda_act = tf.abs(desired_spikes_per_neuron-self.mean_act) #no cost if avg activation is as expectd 
            
            with tf.name_scope('recon_error'):
                self.recon_err = tf.reduce_mean(tf.norm(self.x-self.xp, ord=2, axis=1))

            #calculate cost
            with tf.name_scope("cost_function"):
                #self.lambda_act = tf.reduce_mean(self.activation+1e-5, axis=0)
                self.cost = self.recon_err #+ 1e-3*tf.reduce_mean(self.lambda_act)
 
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
            self.summary_op = tf.summary.merge_all()

        return(mygraph)
    