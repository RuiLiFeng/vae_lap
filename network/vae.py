import tensorflow as tf
from network.abstract_arch import AbstractArch
from network import arch_ops as ops
import numpy as np


class Encoder(AbstractArch):
    def __init__(self, dim_z, hidden_num, exceptions=None, name="Encoder"):
        super(Encoder, self).__init__(name, exceptions)
        self.dim_z = dim_z
        self.hidden_num = hidden_num

    def apply(self, x, is_training=True):
        x = tf.reshape(x, [x.shape[0], -1])
        h = tf.tanh(ops.linear(x, self.hidden_num, scope='hidden'))
        mu = ops.linear(h, self.dim_z, scope='mu')
        log_sigma = ops.linear(h, self.dim_z, scope='sigma_square')
        z = reparametric(mu, log_sigma, 'normal', 'z')
        return mu, log_sigma, z

    def __call__(self, x, is_training, reuse=tf.AUTO_REUSE):
        with tf.variable_scope(self.name, values=[x], reuse=reuse):
            outputs = self.apply(x=x, is_training=is_training)
        return outputs


class Decoder(AbstractArch):
    def __init__(self, img_shape, hidden_num, exceptions=None, name='Decoder'):
        super(Decoder, self).__init__(name, exceptions)
        assert isinstance(img_shape, list), 'img_shape must be list!'
        self.img_shape = img_shape  # [C,H,W]
        self.hidden_num = hidden_num

    def apply(self, z, is_training, flatten=True):
        assert len(z.shape) == 2, "z must be 2-D tensor but found %d-D" % (len(z.shape))
        h = tf.tanh(ops.linear(z, self.hidden_num, scope='hidden'))
        y = tf.nn.sigmoid(ops.linear(h, np.prod(self.img_shape[1:]), scope='y'))
        if not flatten:
            y = tf.reshape(y, [-1] + self.img_shape)
        return y

    def __call__(self, z, is_training, flatten=True, reuse=tf.AUTO_REUSE):
        with tf.variable_scope(self.name, values=[z], reuse=reuse):
            outputs = self.apply(z=z, is_training=is_training, flatten=flatten)
        return outputs


def reparametric(mu, log_sigma, distribution='normal', name=None):
    assert mu.shape == log_sigma.shape, 'The shape of mu and sigma must much, ' \
                                           'found %s and %s' % (mu.shape, log_sigma.shape)
    sigma = tf.exp(log_sigma * 0.5)
    if distribution == 'normal':
        epi = tf.random.normal(mu.shape)
    else:
        raise ValueError('Not supported distribution type %s !' % distribution)
    if name is not None:
        z = tf.add(tf.multiply(epi, sigma), mu, name=name)
    else:
        z = tf.multiply(epi, sigma) + mu
    return z
