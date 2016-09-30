"""Input for the supernet.
N.B. This routine only work on fixed size images dataset.
"""
from __future__ import absolute_import, division, print_function

import scipy.io
import numpy as np
import random
from six.moves import xrange

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
        self._filename_h = [data_dir+prefix_h + str(i) + suffix for i in id_list]
        self._filename_l = [data_dir+prefix_l + str(i) + suffix for i in id_list]
        
        #Read first file for size information
        temp_tensor = np.load(self._filename_h[0])
        [self._n_patch, self._height, self._width] = np.shape(temp_tensor)
                        
        self._epoch = 0        

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
    def epoch(self):
        return self._epoch

    @property
    def n_patch(self):
        return self._n_patch
    
    def batch_generator(self, batch_size):
        """Return data for next batch with a [batch_size, H, W, C] tensor
        Args:
        batch_size: integer
        """
        tensor_next_high = np.zeros([batch_size, self._height, self._width, 1])
        tensor_next_low = np.zeros([batch_size, self._height, self._width, 1])
        
        idf = 0
        while idf < self._n_file:
            ids = list(xrange(self._n_patch))
            random.shuffle(ids)            
            cid = 0
            fnh = self._filename_h[idf]
            fnl = self._filename_l[idf]
            patches_high = np.load(fnh)
            patches_low = np.load(fnl)
            while cid < self._n_patch:
                for pid in xrange(batch_size):                    
                    tensor_next_high[pid, :, :, 0] = patches_high[ids[cid],:,:]
                    tensor_next_low[pid, :, :, 0] = patches_low[ids[cid],:,:]
                    cid += 1
                yield [tensor_next_high, tensor_next_low]
            idf += 1
            if idf == self._n_file:
                self._epoch += 1
                idf = 0
