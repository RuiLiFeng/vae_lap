from __future__ import absolute_import
from __future__ import division
from __future__ import print_function

import functools
import inspect

from absl import logging
import numpy as np
import tensorflow as tf
import tensorflow_datasets as tfds


class ImageDataset(object):
    """
    Interface for Image datasets based on TFDS.

    This method handles data loading settings. The pipline of input operations is as follows:
    1) Shuffle filenames (with seed)
    2) Load file content from disk. Decode images.
    Dataset content after this step is a dictionary.
    3) Prefetch call here.
    4) Filter examples (e.g. by size or label)
    5) Parse example.
    Dataset content after this step is a tuple of tensors (image, label).
    6) Train_only: Repeat dataset.
    7) Transform (random cropping with seed, resizing).
    8) Preprocess ( adding sampled noise/labels with seed).
    Dataset content after this step is a tuple (feature dictionary, label tensor)
    9) Train only: shuffle examples (with seed)
    10) Batch examples.
    11) Prefetch examples.

    step 1-3 are done by load_dataset() and wrap tfds.load()
    step 4-11 are done by train_input_fn() and eval_input_fn()
    """
    def __init__(self, name, tfds_name, tfds_dir, shuffle_buffer_size, cpu_nums,
                 resolution, colors, num_classes, eval_test_samples, seed=547):
        logging.info("ImageDatasetV2(name=%s, tfds_name=%s, resolution=%d, "
                     "colors=%d, num_classes=%s, eval_test_samples=%s, seed=%s)",
                     name, tfds_name, resolution, colors, num_classes,
                     eval_test_samples, seed)
        self.name = name
        self.tfds_name = tfds_name
        self.tfds_dir = tfds_dir
        self.shuffle_buffer_size = shuffle_buffer_size
        self.cpu_nums = cpu_nums
        self.resolution = resolution
        self.colors = colors
        self.num_classes = num_classes
        self.eval_test_samples = eval_test_samples
        self.seed = seed

    @property
    def image_shape(self):
        """Returns a tuple with the image shape."""
        return self.resolution, self.resolution, self.colors

    def parse_fn(self, features):
        image = tf.cast(features['image'], tf.float32) / 255.0
        return image, features['label']

    def load_dataset(self, split):
        """
        Load the underlying dataset split from disk
        :param split:
        :return: tf.data.Dataset object with a tuple of image and label tensor
        """
        ds = tfds.load(
            self.tfds_name,
            split=split,
            data_dir=self.tfds_dir,
            as_dataset_kwargs={'shuffle_files': False}
        )
        ds = ds.map(self.parse_fn, num_parallel_calls=self.cpu_nums)
        return ds.prefetch(tf.contrib.data.AUTOTUNE)

    def train_filter_fn(self, image, label):
        del image, label
        return True

    def train_transform_fn(self, image, label, seed):
        del seed
        return image, label

    def eval_transform_fn(self, image, label, seed):
        del seed
        return image, label

    def train_input_fn(self, batch_size, preprocess_fn=None):
        """Input function for reading data.

            Args:
              batch_size:
              preprocess_fn: Function to process single examples. This is allowed to
                have a `seed` argument.

            Returns:
              `tf.data.Dataset` with preprocessed and batched examples.
            """
        logging.info("train_input_fn(): batch_size=%s seed=%s", batch_size, self.seed)

        ds = self.load_dataset(split=tfds.Split.TRAIN)
        ds = ds.filter(self.train_filter_fn)
        ds = ds.repeat()
        ds = ds.map(functools.partial(self.train_transform_fn, seed=self.seed), num_parallel_calls=self.cpu_nums)
        if preprocess_fn is not None:
            if "seed" in inspect.getargspec(preprocess_fn).args:
                preprocess_fn = functools.partial(preprocess_fn, seed=self.seed)
            ds = ds.map(preprocess_fn, num_parallel_calls=self.cpu_nums)
        ds = ds.shuffle(self.shuffle_buffer_size, seed=self.seed)
        ds = ds.batch(batch_size, drop_remainder=True)
        return ds.prefetch(tf.contrib.data.AUTOTUNE)

    def eval_input_fn(self, batch_size=None, split=None):
        """Input function for reading data.

        Args:
          batch_size
          split: Name of the split to use. If None will use the default eval split
            of the dataset.

        Returns:
          `tf.data.Dataset` with preprocessed and batched examples.
        """
        if split is None:
            split = tfds.Split.TEST
        logging.info("eval_input_fn(): batch_size=%s seed=%s", batch_size, self.seed)

        ds = self.load_dataset(split=split)
        # No filter, no rpeat.
        ds = ds.map(functools.partial(self.eval_transform_fn, seed=self.seed), num_parallel_calls=self.cpu_nums)
        # No shuffle.
        if batch_size is not None:
            ds = ds.batch(batch_size, drop_remainder=True)
        return ds.prefetch(tf.contrib.data.AUTOTUNE)

    def input_fn(self, batch_size, mode='train', preprocess_fn=None):
        assert mode in ['train', 'test', 'eval'], 'Unsupported input_fn mode %s' % mode
        if mode == 'train':
            return self.train_input_fn(batch_size=batch_size, preprocess_fn=preprocess_fn)
        elif mode == 'eval':
            return self.eval_input_fn(batch_size=batch_size)
        else:
            raise ValueError('Unsupported input_fn mode %s' % mode)


class MnistDataset(ImageDataset):
    def __init__(self, tfds_dir, shuffle_buffer_size, cpu_nums, seed):
        super(MnistDataset, self).__init__(
            name='mnist',
            tfds_name='mnist',
            tfds_dir=tfds_dir,
            shuffle_buffer_size=shuffle_buffer_size,
            cpu_nums=cpu_nums,
            resolution=28,
            colors=1,
            num_classes=10,
            eval_test_samples=10000,
            seed=seed
        )


class FashionMnistDataset(ImageDataset):
    """Wrapper for the Fashion-MNIST dataset from TDFS."""
    def __init__(self, tfds_dir, shuffle_buffer_size, cpu_nums, seed):
        super(FashionMnistDataset, self).__init__(
            name="fashion_mnist",
            tfds_name="fashion_mnist",
            tfds_dir=tfds_dir,
            shuffle_buffer_size=shuffle_buffer_size,
            cpu_nums=cpu_nums,
            resolution=28,
            colors=1,
            num_classes=10,
            eval_test_samples=10000,
            seed=seed)


class Cifar10Dataset(ImageDataset):
    """Wrapper for the CIFAR10 dataset from TDFS."""
    def __init__(self, tfds_dir, shuffle_buffer_size, cpu_nums, seed):
        super(Cifar10Dataset, self).__init__(
            name="cifar10",
            tfds_name="cifar10",
            tfds_dir=tfds_dir,
            shuffle_buffer_size=shuffle_buffer_size,
            cpu_nums=cpu_nums,
            resolution=32,
            colors=3,
            num_classes=10,
            eval_test_samples=10000,
            seed=seed)


class CelebaDataset(ImageDataset):
    """Wrapper for the CelebA dataset from TFDS."""
    def __init__(self, tfds_dir, shuffle_buffer_size, cpu_nums, seed):
        super(CelebaDataset, self).__init__(
            name="celeb_a",
            tfds_name="celeb_a",
            tfds_dir=tfds_dir,
            shuffle_buffer_size=shuffle_buffer_size,
            cpu_nums=cpu_nums,
            resolution=64,
            colors=3,
            num_classes=None,
            eval_test_samples=10000,
            seed=seed)

    def parse_fn(self, features):
        """Returns 64x64x3 image and constant label."""
        image = features["image"]
        image = tf.image.resize_image_with_crop_or_pad(image, 160, 160)
        # Note: possibly consider using NumPy's imresize(image, (64, 64))
        image = tf.image.resize_images(image, [64, 64])
        image.set_shape(self.image_shape)
        image = tf.cast(image, tf.float32) / 255.0
        label = tf.constant(0, dtype=tf.int32)
        return image, label


class LsunBedroomDataset(ImageDataset):
    """Wrapper from the LSUN Bedrooms dataset from TFDS."""
    def __init__(self, tfds_dir, shuffle_buffer_size, cpu_nums, seed):
        super(LsunBedroomDataset, self).__init__(
            name="lsun-bedroom",
            tfds_name="lsun/bedroom",
            tfds_dir=tfds_dir,
            shuffle_buffer_size=shuffle_buffer_size,
            cpu_nums=cpu_nums,
            resolution=128,
            colors=3,
            num_classes=None,
            eval_test_samples=30000,
            seed=seed)

    # As the official LSUN validation set only contains 300 samples, which is
    # insufficient for FID computation, we're splitting off some trianing
    # samples. The smallest percentage selectable through TFDS is 1%, so we're
    # going to use that (corresponding roughly to 30000 samples).
    # If you want to use fewer eval samples, just modify eval_test_samples.
    # self._train_split, self._eval_split = \
    #     tfds.Split.TRAIN.subsplit([99, 1])

    def parse_fn(self, features):
        """Returns a 128x128x3 Tensor with constant label 0."""
        image = features["image"]
        image = tf.image.resize_image_with_crop_or_pad(
            image, target_height=128, target_width=128)
        image = tf.cast(image, tf.float32) / 255.0
        label = tf.constant(0, dtype=tf.int32)
        return image, label


DATASETS = {
    "celeb_a": CelebaDataset,
    "cifar10": Cifar10Dataset,
    "fashion-mnist": FashionMnistDataset,
    "lsun-bedroom": LsunBedroomDataset,
    "mnist": MnistDataset,
}


def get_dataset(name, tfds_dir, cpu_nums, shuffle_buffer_size=10000, seed=547):
    """Instantiates a data set and sets the random seed."""
    if name not in DATASETS:
        raise ValueError("Dataset %s is not available." % name)
    return DATASETS[name](tfds_dir, shuffle_buffer_size, cpu_nums, seed=seed)
