import tensorflow as tf
import numpy as np
from utils.utils import *
from datasets.datasets import ReadRecord
import network.vae as vae

import functools
import tensorflow_datasets as tfds
import inspect


def training_loop(config):
    dataset = ReadRecord(config.record_dir + '/Mnist20_rep.tfrecords', [1, 28, 28])
    dataset = dataset.make_initializable_iterator()

