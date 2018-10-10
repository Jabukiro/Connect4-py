from __future__ import absolute_import
from __future__ import division
from __future__ import print_function

import sys
sys.path.insert(0, 'Lib/site-packages/')
import tensorflow as tf
import numpy as np

def in_sum(self):
    buffer = 0
    for w in range(neurons):
        self.neuron[neurons] = tf.accumulate(tf.multiply(self.X, self.W1[neurons], name="nOUT"))
