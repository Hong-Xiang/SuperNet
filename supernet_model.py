from __future__ import absolute_import, division, print_function
import tensorflow as tf
from supernet.supernet_input import DataSet

FLAGS = tf.app.flags.FLAGS

class SuperNet(object):
  """Super resolution net
  """
  def __init__(self):
    pass

  def _indentity_net(self, t_in):
    with tf.variable_scope("indentity_net_output") as scope:
      dtype = tf.float32
      shape = [FLAGS.batch_size, FLAGS.height, FLAGS.width, 1]
      initer = tf.constant_initializer(0.0, dtype)
      t_out = tf.get_variable("output_image", shape, dtype, initer)
    with tf.name_scope("indentity_net") as scope:
      t_out.assign(t_in)
    return t_out

  def inference(self, low_resolution_images):
    high_resolution_images = self._indentity_net(low_resolution_images)
    return high_resolution_images

  def _l2norm_loss(inference_images, reference_images):
    pass

  def loss(self, inference_images, high_resolution_images):
    pass
