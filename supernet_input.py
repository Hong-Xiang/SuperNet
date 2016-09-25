"""Input for the supernet.
N.B. This routine only work on fixed size images dataset.
"""
from __future__ import absolute_import, division, print_function

import scipy.io
import numpy as np

def load_images(filename):  
  mat_data = scipy.io.loadmat(filename)  
  tensor_images =  np.array(mat_data.get("data"))
  return tensor_images

class DataSet(object):
  def __init__(self, data_dir, prefix_h, prefix_l, id_list, suffix):
    """Constructor of DataSet object.
    Args:
    data_dir: string, path to dataset, end with '/'
    prefix_h: prefix of high resolution image,
    prefix_l: prefix of low resolution image,
    id_list: list of id of image files,
    suffix: suffix of image files.    
    """
    self._n_file = len(id_list)    
    self._filename_h = [data_dir+prefix_h + str(id) + suffix for id in id_list]
    self._filename_l = [data_dir+prefix_l + str(id) + suffix for id in id_list]

    #Read first file for size information
    temp_tensor = load_images(self._filename_h[0])
    [self._n_image_per_file, self._height, self._width] = np.shape(temp_tensor)

    self._n_image_left = 0
    self._id_next_file = 0    

    self._high_buffer = np.zeros([self._n_image_per_file, self._height, self._width, 1])
    self._low_buffer = np.zeros([self._n_image_per_file, self._height, self._width, 1])

    self._epoch = 0
    pass

  @property
  def n_files(self):
    return self._n_file

  @property
  def files_high(self):    
    return self._filename_h

  @property
  def files_low(self):
    return self._filename_l

  @property
  def height(self):
    return self._height

  @property
  def width(self):
    return self._width

  @property
  def n_image_left(self):
    return self._n_image_left

  def epoch(self):
    return self._epoch

  def _read_next_file(self):
    #load images
    tensor_high = load_images( self._filename_h[self._id_next_file] )
    tensor_low = load_images( self._filename_l[self._id_next_file] )

    #update buffer
    self._high_buffer[:,:,:,0] = tensor_high
    self._low_buffer[:,:,:,0] = tensor_low    
    self._n_image_left = self._n_image_per_file
      
    #update next file id
    self._id_next_file = self._id_next_file + 1
    if self._id_next_file == self._n_file : 
      self._id_next_file = 0
      self._epoch += 1         

  def next_batch(self, batch_size):
    """Return data for next batch with a [batch_size, H, W, C] tensor
    Args:
    batch_size: integer
    """
    tensor_next_high = np.zeros([batch_size, self._height, self._width, 1])
    tensor_next_low = np.zeros([batch_size, self._height, self._width, 1])

    to_out = batch_size #number of images(tensor) to output
    do_out = 0 #number of images(tensor) done output(put into tensor)

    #while more file is required
    while to_out > self._n_image_left :
      #dump out all read.
      id_valid_buffer = self._n_image_per_file - self._n_image_left
      o_ids = list(xrange(do_out, do_out + self._n_image_left - 1))
      b_ids = list(xrange(id_valid_buffer, self._n_image_per_file - 1))
      tensor_next_high[o_ids, :, :, 0] = self._high_buffer[b_ids, :, :, 0]
      tensor_next_low[o_ids, :, :, 0] = self._low_buffer[b_ids, :, :, 0]
      to_out = to_out - self._n_image_left
      do_out = do_out + self._n_image_left

      #read next file
      self._read_next_file()
                                
    #while loaded images is enough
    id_valid_buffer = self._n_image_per_file - self._n_image_left
    o_ids = list(xrange(do_out, to_out))
    b_ids = list(xrange(id_valid_buffer, id_valid_buffer+to_out))
    tensor_next_high[o_ids, :, :, 0] = self._high_buffer[b_ids, :, :, 0]
    tensor_next_low[o_ids, :, :, 0] = self._low_buffer[b_ids, :, :, 0]
    self._n_image_left -= to_out

    return tensor_next_high, tensor_next_low    

