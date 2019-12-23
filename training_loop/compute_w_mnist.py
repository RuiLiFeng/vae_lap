import tensorflow as tf
import numpy as np
from utils.utils import *
from datasets.datasets import get_dataset
import network.vae as vae

import functools
import tensorflow_datasets as tfds
import inspect


EPS = 1e-10

# compute the representation and sigma based on given datasets and auto encoder


def training_loop(config: Config):
    timer = Timer()
    print('Task name %s' % config.task_name)
    print('Loading %s dataset...' % config.dataset_name)
    dset = get_dataset(config.dataset_name, config.tfds_dir, config.gpu_nums * 2)
    dataset = train_input_fn(dset, config.batch_size)
    dataset = dataset.make_initializable_iterator()
    print("Constructing networks...")
    Encoder = vae.Encoder(config.dim_z, config.e_hidden_num, exceptions=['opt'], name='VAE_En')
    Decoder = vae.Decoder(config.img_shape, config.d_hidden_num, exceptions=['opt'], name='VAE_De')
    print("Building tensorflow graph...")
    image, label = dataset.get_next()
    _, _, z = Encoder(image, is_training=True)
    sigma2_plus = compute_sigma2(z)
    print("Building eval module...")

    fixed_z = tf.constant(np.random.normal(size=[config.example_nums, config.dim_z]), dtype=tf.float32)
    fixed_z0 = tf.constant(np.random.normal(size=[config.example_nums, config.dim_z]), dtype=tf.float32)
    fixed_z1 = tf.constant(np.random.normal(size=[config.example_nums, config.dim_z]), dtype=tf.float32)
    fixed_x = tf.placeholder(tf.float32, [config.example_nums] + config.img_shape)
    fixed_x0 = tf.placeholder(tf.float32, [config.example_nums] + config.img_shape)
    fixed_x1 = tf.placeholder(tf.float32, [config.example_nums] + config.img_shape)
    input_dict = {'fixed_z': fixed_z, 'fixed_z0': fixed_z0, 'fixed_z1': fixed_z1, 'fixed_x': fixed_x,
                  'fixed_x0': fixed_x0, 'fixed_x1': fixed_x1, 'num_midpoints': config.num_midpoints}

    def sample_step():
        out_dict = generate_sample(Decoder, input_dict)
        out_dict.update(reconstruction_sample(Encoder, Decoder, input_dict))
        out_dict.update({'fixed_x': fixed_x, 'fixed_x0': fixed_x0, 'fixed_x1': fixed_x1})
        return out_dict

    o_dict = sample_step()

    print("Building init module...")
    with tf.init_scope():
        init = [tf.global_variables_initializer(), dataset.initializer]
        saver_e = tf.train.Saver(Encoder.restore_variables)
        saver_d = tf.train.Saver(Decoder.restore_variables)

    print('Starting training...')
    with tf.Session(config=tf.ConfigProto(allow_soft_placement=True)) as sess:
        sess.run(init)
        if config.resume:
            print("Restore vae...")
            saver_e.restore(sess, config.restore_e_dir)
            saver_d.restore(sess, config.restore_d_dir)
        timer.update()
        print('Preparing sample utils...')

        fixed_x_, _ = get_fixed_x(sess, dataset, config.example_nums, config.batch_size)
        fixed_x0_, _ = get_fixed_x(sess, dataset, config.example_nums, config.batch_size)
        fixed_x1_, _ = get_fixed_x(sess, dataset, config.example_nums, config.batch_size)
        o_dict_ = sess.run(o_dict, {fixed_x: fixed_x_, fixed_x0: fixed_x0_, fixed_x1: fixed_x1_})
        for key in o_dict_:
            save_image_grid(o_dict_[key], config.model_dir + '/%s.jpg' % key)
        print("Completing all work, iteration now start, consuming %s " % timer.runing_time_format)

        print("Start iterations...")
        sigma2 = 0.0
        count = 0
        with tf.io.TFRecordWriter(config.model_dir + '/Mnist20_rep.tfrecords') as writer:
            while True:
                try:
                    image_, label_, sigma2_plus_, rep_ = sess.run([image, label, sigma2_plus, z])
                    sigma2 += sigma2_plus_
                    count += 1
                    for n in range(image_.shape[0]):
                        tf_example = serialize_example(image_[n], label_[n], rep_[n])
                        writer.write(tf_example)
                    if count % 100 == 0:
                        timer.update()
                        print('Complete %d bathes, consuming time %s' % (count, timer.runing_time_format))
                except tf.errors.OutOfRangeError:
                    np.save(config.model_dir + '/sigma2.npy', sigma2 / (count * config.batch_size))
                    print('Done!')
                    break


def get_fixed_x(sess, dataset, num, batch_size):
    num_batch, res = divmod(num, batch_size)
    xs = []
    for i in range(num_batch + 1):
        if i < num_batch:
            xs.append(sess.run(dataset.get_next()[0]))
        else:
            xs.append(sess.run(dataset.get_next()[0])[:res])
    x = np.concatenate(xs, 0)
    return x, None


def compute_sigma2(z):
    y1_ = tf.tile(tf.expand_dims(z, 1), [1, z.shape[0].value, 1])
    y2_ = tf.tile(tf.expand_dims(z, 0), [z.shape[0].value, 1, 1])
    pairwise_dis = tf.reduce_sum(tf.square(y1_ - y2_), axis=2)
    return tf.reduce_sum(pairwise_dis)


def _bytes_feature(value):
    """Returns a bytes_list from a string / byte."""
    if isinstance(value, type(tf.constant(0))):
        value = value.numpy() # BytesList won't unpack a string from an EagerTensor.
    return tf.train.Feature(bytes_list=tf.train.BytesList(value=[value]))


def _float_feature(value):
    """Returns a float_list from a float / double."""
    return tf.train.Feature(float_list=tf.train.FloatList(value=[value]))


def _int64_feature(value):
    """Returns an int64_list from a bool / enum / int / uint."""
    return tf.train.Feature(int64_list=tf.train.Int64List(value=[value]))


def serialize_example(img, label, representation):
    feature = {
        'image': _bytes_feature(img.tostring()),
        'label': _int64_feature(label),
        'representation': _bytes_feature(representation.tostring())
    }
    example_proto = tf.train.Example(features=tf.train.Features(feature=feature))
    return example_proto.SerializeToString()


def train_input_fn(dset, batch_size, preprocess_fn=None):
    """Input function for reading data.

        Args:
          batch_size:
          preprocess_fn: Function to process single examples. This is allowed to
            have a `seed` argument.

        Returns:
          `tf.data.Dataset` with preprocessed and batched examples.
        """
    ds = dset.load_dataset(split=tfds.Split.TRAIN)
    ds = ds.filter(dset.train_filter_fn)
    ds = ds.map(functools.partial(dset.train_transform_fn, seed=dset.seed), num_parallel_calls=dset.cpu_nums)
    if preprocess_fn is not None:
        if "seed" in inspect.getargspec(preprocess_fn).args:
            preprocess_fn = functools.partial(preprocess_fn, seed=dset.seed)
        ds = ds.map(preprocess_fn, num_parallel_calls=dset.cpu_nums)
    ds = ds.shuffle(dset.shuffle_buffer_size, seed=dset.seed)
    ds = ds.batch(batch_size, drop_remainder=True)
    return ds.prefetch(tf.contrib.data.AUTOTUNE)


