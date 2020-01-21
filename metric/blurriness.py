import tensorflow as tf
import numpy as np


class blurr(object):
    def __init__(self, data_shape): #[n,h,w,c]
        self.data_shape = data_shape
        self.lap_filter = np.array([[0, 1, 0], [1, -4, 1], [0, 1, 0]])
        self.lap_filter = self.lap_filter.reshape([3, 3, 1, 1])

    def __call__(self, x):
        if self.data_shape[-1] > 1:
            x = tf.image.rgb_to_grayscale(x)
        conv = tf.nn.conv2d(x, self.lap_filter, strides=[1, 1, 1, 1], padding='VALID')
        _, lapvar = tf.nn.moments(conv, axes=[1, 2, 3])
        return lapvar
