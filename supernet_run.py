from __future__ import absolute_import, division, print_function

import tensorflow as tf
import time
import numpy as np

from six.moves import xrange

FLAGS = tf.app.flags.FLAGS

TEST_NAME = "Patch-Sino"

def define_flags():
    flag = tf.app.flags
    #flag.DEFINE_string("data_dir","/home/hongxwing/Workspace/Datas/flickr25k_with_crop/","Path to data directory")
    flag.DEFINE_string("data_dir","/home/hongxwing/Workspace/Datas/SinogramPatchData/","Path to data directory")
    flag.DEFINE_string("prefix_h","patch_high_","Prefix of high resolution data filename.")
    flag.DEFINE_string("prefix_l","patch_low_","Prefix of low resoution data filename.")
    flag.DEFINE_string("suffix",".npy", "Suffix of data filename.")
    flag.DEFINE_integer("batch_size", 256, "Batch size of train dataset.")
    flag.DEFINE_integer("height", None, "Height of images.")
    flag.DEFINE_integer("width", None, "Width of images.")

    flag.DEFINE_string("summaries_dir","./summary/"+TEST_NAME,"Path to summary directory.")
    flag.DEFINE_integer("conv_height", 3, "Convolution window height.")
    flag.DEFINE_integer("conv_width", 3, "Convolution window width.")
    flag.DEFINE_integer("output_conv_height", 3, "Convolution window height of output layer.")
    flag.DEFINE_integer("output_conv_width", 3, "Convolution window width of output layer.")
    flag.DEFINE_float("stddev", 2e-2, "Default std dev of weight variable.")
    flag.DEFINE_float("learning_rate_init", 1e-3, "Initial learning rate.")
    flag.DEFINE_float("learning_rate_decay_factor",0.6,"Learning rate decay factor.")
    
    flag.DEFINE_integer("decay_epoch", 1, "Decay epoch.")
    flag.DEFINE_integer("decay_steps", None, "Decay steps.")
    
    flag.DEFINE_integer("max_step", 5000, "Max train steps.")
define_flags() 


from supernet.supernet_model import SuperNet
from supernet.supernet_input import DataSet


def main(args):
    data = DataSet(FLAGS.data_dir, FLAGS.prefix_h, FLAGS.prefix_l, list(xrange(0,4)), FLAGS.suffix)
    FLAGS.height = data.height
    FLAGS.width = data.width  
    #FLAGS.decay_steps = data.n_files*FLAGS.decay_epoch*data.n_image_per_file//FLAGS.batch_size+1
    FLAGS.decay_steps = 300
    sn = SuperNet()  
    loss = sn.loss
    train = sn.train_step
    merged = tf.merge_all_summaries()
    sess = tf.Session()
    init = tf.initialize_all_variables()
    
    saver = tf.train.Saver(tf.all_variables())
    
    summary_writer = tf.train.SummaryWriter(FLAGS.summaries_dir, sess.graph)
    
    sess.run(init)
    RESTORE = False
    if RESTORE:
        saver.restore(sess, "./supernet-"+TEST_NAME+"-"+str(FLAGS.max_step))    
    
    pg = data.batch_generator(FLAGS.batch_size)
    # [high_res_data, low_res_data] = pg.next()
    # [res_i, res_r, inf_img, mid_res, loss_v, _] = sess.run([sn.residual_inference, sn.residual_reference, sn.inference, sn.mid_results, loss, train], \
    #         feed_dict = {sn.low_res_images: low_res_data, sn.high_res_images: high_res_data})
    # print(loss_v)
    # np.save('res_i.npy',res_i)
    # np.save('res_r.npy',res_r)
    # np.save('inf.npy',inf_img)
    # cid = 0
    # for ts in mid_res:
    #     np.save('mid'+str(cid)+'.npy',ts)
    #     cid += 1
    # np.save('high_res.npy', high_res_data)
    # np.save('low_res.npy',low_res_data)    
    
    for i in xrange(FLAGS.max_step):
        [high_res_data, low_res_data] = pg.next()
        residual = high_res_data - low_res_data   
        #summary, loss_v, _ = sess.run([merged, loss, train], feed_dict={sn.low_res_images: low_res_data, sn.high_res_images: high_res_data})
        summary, loss_v, _ = sess.run([merged, loss, train], feed_dict={sn.low_res_images: low_res_data, sn.residual_reference: residual})
        print("step = " + str(i) + ", loss = ", str(loss_v))
        if(i%10==0):            
            summary_writer.add_summary(summary, i)                
    saver.save(sess, 'supernet-'+TEST_NAME, global_step=FLAGS.max_step)
if __name__ == "__main__":
    tf.app.run()