from __future__ import absolute_import, division, print_function
import tensorflow as tf
from supernet.supernet_input import DataSet

import supernet.supernet_utils as ut

FLAGS = tf.app.flags.FLAGS

class SuperNet(object):
    """Super resolution net
    """
    def __init__(self):
        self._global_step = tf.Variable(0, trainable = False, name = 'global_step')        
        
        self._mid_result = list()
        
        self._low_res_images = ut.input_layer([FLAGS.batch_size, FLAGS.height, FLAGS.width, 1],"input_low_res")
        #self._high_res_images = ut.label_layer([FLAGS.batch_size, FLAGS.height, FLAGS.width, 1],"label_high_res")
        #self._residual_reference = tf.sub(tensor_high_res, tensor_low_res, name = "residual_reference")        
        self._residual_reference = ut.label_layer([FLAGS.batch_size, FLAGS.height, FLAGS.width, 1],"residual_reference")
        self._high_res_images = tf.add(self._low_res_images, self._residual_reference,"label_high_res")

        N_CHANNEL = 128
        c1 = ut.conv_layer(self._low_res_images, [3, 3, 1, N_CHANNEL], name = "conv_feature_1")                
        c2 = ut.conv_layer(c1, [3, 3, N_CHANNEL, N_CHANNEL], name = "conv_feature_2")
        c3 = ut.conv_layer(c2, [3, 3, N_CHANNEL, N_CHANNEL], name = "conv_feature_3")
        c4 = ut.conv_layer(c3, [3, 3, N_CHANNEL, N_CHANNEL], name = "conv_feature_4")
        c5 = ut.conv_layer(c4, [3, 3, N_CHANNEL, N_CHANNEL], name = "conv_feature_5")
        c6 = ut.conv_layer(c5, [3, 3, N_CHANNEL, N_CHANNEL], name = "conv_feature_6")      
        c7 = ut.conv_layer(c6, [3, 3, N_CHANNEL, N_CHANNEL], name = "conv_feature_7")

        c8 = ut.conv_layer(c7, [1, 1, N_CHANNEL, N_CHANNEL], name = "full_connect")
        
        c9 = ut.conv_layer(c8, [3, 3, N_CHANNEL, N_CHANNEL], name = "conv_recon_9")
        c10 = ut.conv_layer(c9, [3, 3, N_CHANNEL, N_CHANNEL], name = "conv_recon_10")
        c11 = ut.conv_layer(c10, [3, 3, N_CHANNEL, N_CHANNEL], name = "conv_recon_11")
        self._residual_inference = ut.output_layer(c11, [1, 1, N_CHANNEL, 1], name = "output")
        
        
        self._loss = ut.l2_loss_layer(self._residual_reference, self._residual_inference, name = "loss")        
                                                                     
        self._inference_image = tf.add(self._low_res_images, self._residual_inference, name = "image_inference")                        

        # Decay the learning rate exponentially based on the number of steps.
        with tf.name_scope("train") as scope:
            lr = tf.train.exponential_decay(FLAGS.learning_rate_init,
                                            self._global_step,
                                            FLAGS.decay_steps,
                                            FLAGS.learning_rate_decay_factor,
                                            staircase=True, 
                                            name = scope+'/learning_rate')                                    
            tf.scalar_summary("learning rate", lr)
            self._train_step = tf.train.AdamOptimizer(lr).minimize(self._loss, 
                                                                   self._global_step, 
                                                                   name=scope+'/train_step')
            #self._train_step = tf.train.GradientDescentOptimizer(lr).minimize(loss,global_step=step)  
        
        tf.image_summary('input_low',self._low_res_images)
        tf.image_summary('label_high',self._high_res_images)
        tf.image_summary('residual_reference',self._residual_reference)
        tf.image_summary('residual_inference',self._residual_inference)
        tf.image_summary('inference',self._inference_image)
    
    @property
    def inference(self):
        return self._inference_image
    
    @property
    def residual_inference(self):
        return self._residual_inference

    @property
    def residual_reference(self):
        return self._residual_reference

    @property
    def loss(self):
        return self._loss

    @property
    def low_res_images(self):
        return self._low_res_images

    @property
    def high_res_images(self):
        return self._high_res_images
    
    @property
    def train_step(self):
        return self._train_step

    @property
    def mid_results(self):
        return self._mid_result
