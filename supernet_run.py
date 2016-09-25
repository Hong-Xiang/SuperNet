from __future__ import absolute_import, division, print_function

import tensorflow as tf

from supernet.supernet_model import SuperNet
from supernet.supernet_input import DataSet
#define flags:



FLAGS = tf.app.flags.FLAGS

def _define_flags():
  flag = tf.app.flags
  flag.DEFINE_string("data_dir","/home/hongxwing/Workspace/Datas/flickr25k_processed/","Path to data directory")
  flag.DEFINE_string("prefix_h","imh","Prefix of high resolution data filename.")
  flag.DEFINE_string("prefix_l","iml","Prefix of low resoution data filename.")
  flag.DEFINE_string("suffix",".mat", "Suffix of data filename.")
  flag.DEFINE_integer("batch_size", 32, "Batch size of train dataset.")
  flag.DEFINE_integer("height", None, "Height of images.")
  flag.DEFINE_integer("width", None, "Width of images.")
  flag.DEFINE_string("summaries_dir","./summary/","Path to summary directory.")

def main(args):  
  data = DataSet(FLAGS.data_dir, FLAGS.prefix_h, FLAGS.prefix_l, [0], FLAGS.suffix)
  FLAGS.height = data.height
  FLAGS.width = data.width  
  
  sn = SuperNet()
  dtype = tf.float32
  shape = [None, 256, 256, 1]
  high_res_imgs_net = tf.placeholder(dtype, shape, "train_high_res_imgs")
  low_res_imgs_net = tf.placeholder(dtype, shape, "train_low_res_imgs")
  test = sn.inference(low_res_imgs_net)
  sess = tf.Session()
  high_res_data, low_res_data = data.next_batch(FLAGS.batch_size)
  init = tf.initialize_all_variables()
  sess.run(init)
  sess.run(test, feed_dict={low_res_imgs_net: low_res_data})

if __name__ == "__main__":
  _define_flags()
  tf.app.run()