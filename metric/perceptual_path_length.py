import numpy as np
import tensorflow as tf


# Normalize batch of vectors.
def normalize(v):
    return v / tf.sqrt(tf.reduce_sum(tf.square(v), axis=-1, keepdims=True))


# Spherical interpolation of a batch of vectors.
def slerp(a, b, t):
    a = normalize(a)
    b = normalize(b)
    d = tf.reduce_sum(a * b, axis=-1, keepdims=True)
    p = t * tf.math.acos(d)
    c = normalize(b - d * a)
    d = a * tf.math.cos(p) + c * tf.math.sin(p)
    return normalize(d)


# Linear interpolation of a batch of vectors.
def lerp(x0, x1, t):
    return x0 + (x1 - x0) * t


class PPL(object):
    def __init__(self, num_samples, epsilon, x_shape, y_shape, sampling):
        assert sampling in ['full', 'end']
        self.sampling = sampling
        self.x_shape = x_shape
        self.y_shape = y_shape
        self.num_samples = num_samples
        self.epsilon = epsilon
        self.vgg = tf.keras.applications.VGG16(input_shape=self.y_shape, include_top=False, weights='imagenet')
        self.vgg.trainable = False

    def apply(self, G, x0, x1):
        lerp_t = tf.random_uniform([self.x_shape[0].value], 0.0, 1.0 if self.sampling == 'full' else 0.0)
        d_x0 = lerp(x0, x1, lerp_t[:, np.newaxis, np.newaxis])
        d_x1 = lerp(x0, x1, lerp_t[:, np.newaxis, np.newaxis] + self.epsilon)
        d_y0 = G(d_x0)
        d_y1 = G(d_x1)

        # crop only the digits region
        c = int(d_y0.shape[2] // 8)
        d_y0 = d_y0[:, c * 3: c * 7, c * 2: c * 6, :]
        d_y1 = d_y1[:, c * 3: c * 7, c * 2: c * 6, :]

        # scale dynamic range
        d_y0 = d_y0 * 255
        d_y1 = d_y1 * 255

        if d_y0.shape[-1] == 1:
            d_y0 = tf.image.grayscale_to_rgb(d_y0)
            d_y1 = tf.image.grayscale_to_rgb(d_y1)
        d_y0_logits = self.vgg(d_y0)
        d_y1_logits = self.vgg(d_y1)

        distance = tf.reduce_sum(tf.square(d_y0_logits - d_y1_logits), axis=[1, 2, 3])
        distance = tf.reduce_mean(distance)
        return distance

    def __call__(self, G, x0, x1):
        return self.apply(G, x0, x1)


class PPL_mnist(object):
    def __init__(self, epsilon, sampling, generator, d_metric):
        assert sampling in ['full', 'end']
        self.sampling = sampling
        self.epsilon = epsilon
        self.d_metric = d_metric
        self.generator = generator

    def apply(self, x0, x1):
        lerp_t = tf.random_uniform([x0.shape[0].value], 0.0, 1.0 if self.sampling == 'full' else 0.0)
        dims = len(list(x0.shape)) - 1
        if dims == 1:
            lerp_t = tf.expand_dims(lerp_t, 1)
        elif dims == 2:
            lerp_t = tf.expand_dims(tf.expand_dims(lerp_t, 1), 2)
        elif dims == 3:
            lerp_t = tf.expand_dims(tf.expand_dims(tf.expand_dims(lerp_t, 1), 2), 3)
        else:
            raise ValueError("x0 dims must not exceed 4 but found %d" % x0.shape)
        d_x0 = lerp(x0, x1, lerp_t)
        d_x1 = lerp(x0, x1, lerp_t + self.epsilon)
        d_y0 = self.generator(d_x0)
        d_y1 = self.generator(d_x1)

        d_y0_logits = self.d_metric(d_y0)
        d_y1_logits = self.d_metric(d_y1)
        distance = tf.reduce_sum(tf.square(d_y0_logits - d_y1_logits), axis=1) / self.epsilon ** 2
        distance = tf.reduce_mean(distance)
        return distance

    def __call__(self, x0, x1):
        return self.apply(x0, x1)
