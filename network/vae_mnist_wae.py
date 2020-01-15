import tensorflow as tf
from network.abstract_arch import AbstractArch
from network import arch_ops as ops
import numpy as np


class Encoder(AbstractArch):
    def __init__(self, dim_z=64, nef=1024, nel=4, exceptions=None, name="Encoder"):
        super(Encoder, self).__init__(exceptions=exceptions, name=name)
        self.dim_z = dim_z
        self.nef = nef
        self.nel = nel

    def apply(self, x, is_training):
        num_units = self.nef
        num_layers = self.nel
        for i in range(num_layers):
            scale = 2 ** (num_layers - i - 1)
            x = ops.conv2d(x, num_units // scale, k_w=5, k_h=5, d_h=2, d_w=2, stddev=0.0099999, name='h%d_conv' % i)
            x = ops.batch_norm(x, is_training, name='h%d_bn' % i)
            x = tf.nn.relu(x)
        x = tf.reshape(x, [x.shape[0], -1])
        mu = ops.linear(x, self.dim_z, scope='mu')
        log_sigma = ops.linear(x, self.dim_z, scope='log_sigma')
        return mu, log_sigma, reparametric(mu, log_sigma)

    def __call__(self, x, is_training, reuse=tf.AUTO_REUSE):
        with tf.variable_scope(self.name, values=[x], reuse=reuse):
            outputs = self.apply(x=x, is_training=is_training)
        return outputs


class Decoder(AbstractArch):
    def __init__(self, img_shape, ndf=1024, ndl=3, exceptions=None, name="Encoder"):
        super(Decoder, self).__init__(exceptions=exceptions, name=name)
        self.img_shape = img_shape
        self.ndl = ndl
        self.ndf = ndf

    def apply(self, z, is_training, flatten=False):
        output_shape = self.img_shape
        num_units = self.ndf
        batch_size = tf.shape(z)[0]
        num_layers = self.ndl
        height = output_shape[0] // 2**(num_layers - 1)
        width = output_shape[1] // 2**(num_layers - 1)
        z = ops.linear(z, num_units * height * width)
        z = tf.reshape(z, [-1, height, width, num_units])
        z = tf.nn.relu(z)
        for i in range(num_layers - 1):
            scale = 2 ** (i + 1)
            _out_shape = [batch_size, height * scale,
                          width * scale, num_units // scale]
            z = deconv2d(z, _out_shape, stddev=0.0099999, conv_filters_dim=5, scope='h%d_deconv' % i)
            z = ops.batch_norm(z, is_training, name='h%d_bn' % i)
            z = tf.nn.relu(z)
        _out_shape = [batch_size] + list(output_shape)
        z = deconv2d(z, _out_shape, stddev=0.0099999, d_h=1, d_w=1, conv_filters_dim=5, scope='d_out')
        return tf.nn.tanh(z)

    def __call__(self, z, is_training, flatten=False, reuse=tf.AUTO_REUSE):
        with tf.variable_scope(self.name, values=[z], reuse=reuse):
            outputs = self.apply(z=z, is_training=is_training, flatten=flatten)
        return outputs


def deconv2d(input_, output_shape, d_h=2, d_w=2, stddev=0.02, scope=None, conv_filters_dim=None, padding='SAME'):
    """Transposed convolution (fractional stride convolution) layer.
    """

    shape = input_.get_shape().as_list()
    k_h = conv_filters_dim
    k_w = k_h

    assert len(shape) == 4, 'Conv2d_transpose works only with 4d tensors.'
    assert len(output_shape) == 4, 'outut_shape should be 4dimensional'

    with tf.variable_scope(scope or "deconv2d"):
        w = tf.get_variable(
            'filter', [k_h, k_w, output_shape[-1], shape[-1]],
            initializer=tf.random_normal_initializer(stddev=stddev))
        deconv = tf.nn.conv2d_transpose(
            input_, w, output_shape=output_shape,
            strides=[1, d_h, d_w, 1], padding=padding)
        biases = tf.get_variable(
            'b', [output_shape[-1]],
            initializer=tf.constant_initializer(0.0))
        deconv = tf.nn.bias_add(deconv, biases)
    return deconv


def reparametric(mu, log_sigma, distribution='normal', name=None):
    assert mu.shape == log_sigma.shape, 'The shape of mu and sigma must much, ' \
                                           'found %s and %s' % (mu.shape, log_sigma.shape)
    sigma = tf.exp(log_sigma * 0.5)
    if distribution == 'normal':
        epi = tf.random.normal(mu.shape, dtype=mu.dtype)
    else:
        raise ValueError('Not supported distribution type %s !' % distribution)
    if name is not None:
        z = tf.add(tf.multiply(epi, sigma), mu, name=name)
    else:
        z = tf.multiply(epi, sigma) + mu
    return z


class Discriminator(AbstractArch):
    def __init__(self, dim_z, hidden_num=512, num_layer=4, pz_scale=1.0, nowozin_trick=False,
                 exceptions=None, name="Discriminator"):
        super(Discriminator, self).__init__(name, exceptions)
        self.dim_z = dim_z
        self.hidden_num = hidden_num
        self.num_layer = num_layer
        self.nowozin_trick = nowozin_trick
        self.pz_scale = pz_scale

    def apply(self, z, is_training=True):
        hi = z
        for i in range(self.num_layer):
            hi = ops.linear(hi, self.hidden_num, scope='hidden%d_linear' % i)
            hi = tf.nn.relu(hi)
        hi = ops.linear(hi, 1, scope='final_linear')
        if self.nowozin_trick:
            sigma2_p = float(self.pz_scale) ** 2
            normsq = tf.reduce_sum(tf.square(z), 1)
            hi = hi - normsq / 2. / sigma2_p - 0.5 * tf.log(2. * np.pi) - 0.5 * self.dim_z * np.log(sigma2_p)
        return hi

    def __call__(self, z, is_training, reuse=tf.AUTO_REUSE):
        with tf.variable_scope(self.name, values=[z], reuse=reuse):
            outputs = self.apply(z=z, is_training=is_training)
        return outputs
