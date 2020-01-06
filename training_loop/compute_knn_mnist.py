import tensorflow as tf
import numpy as np
from utils.utils import *
from datasets.datasets import ReadRecord
import network.vae as vae

import functools
import tensorflow_datasets as tfds
import inspect
from tqdm import tqdm


def training_loop(config):
    # dset = ReadRecord(config.record_dir + '/Mnist20_rep.tfrecords', [1, 28, 28])
    # dataset = dset.batch(256, True)
    # dataset = dataset.make_initializable_iterator()
    # img, rep, label = dataset.get_next()
    # rep_ = []
    # timer = Timer()
    # with tf.Session() as sess:
    #     sess.run(dataset.initializer)
    #     counter = 0
    #     timer.update()
    #     while True:
    #         try:
    #             rep_.append(sess.run(rep))
    #             counter += 1
    #             if counter % 100 == 0:
    #                 print('Complete 100 batches with %s' % timer.runing_time_format)
    #         except tf.errors.OutOfRangeError:
    #             rep_ = np.concatenate(rep_, 0)
    #             break
    # print(rep_.shape)
    reps = get_rep(config)
    num_img = reps.shape[0]
    # neighbour = []
    n = 5
    K = int(n * np.floor(np.log(num_img)))
    print('K: %d' % K)
    sigma2 = 0.0

    # for i in tqdm(range(num_img)):
    #     cur_rep = rep_[i]
    #     pair_dis = np.repeat(np.expand_dims(cur_rep, 0), num_img, 0)
    #     pair_dis = np.sum(np.square(pair_dis - rep_), 1)
    #     index = np.argsort(pair_dis)[:K]
    #     sigma2 += np.sum(pair_dis[index]) / K
    #     neighbour.append(np.expand_dims(index, 0))
    # neighbour = np.concatenate(neighbour, 0)
    # sigma2 = sigma2 / num_img
    # print(neighbour.shape)
    dset = ReadRecord(config.record_dir + '/Mnist20_rep.tfrecords', [1, 28, 28])
    dataset = dset.batch(256, True).make_initializable_iterator()
    img, rep, label = dataset.get_next()
    with tf.io.TFRecordWriter(config.model_dir + '/Mnist20knn%d_rep.tfrecords' % n) as writer:
        with tf.Session() as sess:
            sess.run(dataset.initializer)
            for i in tqdm(range(num_img // 256)):
                image_, rep_, label_ = sess.run([img, rep, label])
                for j in range(256):
                    pair_dis = np.repeat(np.expand_dims(rep_[j], 0), num_img, 0)
                    pair_dis = np.sum(np.square(pair_dis - reps), 1)
                    index = np.argsort(pair_dis)[:K]
                    sigma2 += np.sum(pair_dis[index]) / K
                    if i == 0 and j == 0:
                        np.save('pair_dis', pair_dis)
                        np.save('index', index)
                        np.save('rep', reps)
                    tf_example = serialize_example(image_[j], label_[j], rep_[j],
                                                   index, i * 256 + j)
                    writer.write(tf_example)
                    del pair_dis, index, tf_example
    np.save(config.model_dir + '/knn%dsigma2.npy' % n, sigma2 / num_img)
    print('sigma %f' % (sigma2 / num_img))


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


def serialize_example(img, label, representation, neighbour, index):
    feature = {
        'image': _bytes_feature(img.tostring()),
        'label': _int64_feature(label),
        'representation': _bytes_feature(representation.tostring()),
        'neighbour': _bytes_feature(neighbour.tostring()),
        'index': _int64_feature(index)
    }
    example_proto = tf.train.Example(features=tf.train.Features(feature=feature))
    return example_proto.SerializeToString()


def get_rep(config):
    dset = ReadRecord(config.record_dir + '/Mnist20_rep.tfrecords', [1, 28, 28])
    dataset = dset.batch(256, True)
    dataset = dataset.make_initializable_iterator()
    img, rep, label = dataset.get_next()
    rep_ = []
    timer = Timer()
    with tf.Session() as sess:
        sess.run(dataset.initializer)
        counter = 0
        timer.update()
        while True:
            try:
                rep_.append(sess.run(rep))
                counter += 1
                if counter % 100 == 0:
                    print('Complete 100 batches with %s' % timer.runing_time_format)
            except tf.errors.OutOfRangeError:
                rep_ = np.concatenate(rep_, 0)
                break
    print(rep_.shape)
    del dataset, dset, img, rep, label, timer
    return rep_
