import csv
from argparse import ArgumentParser
import tensorflow as tf


usage = 'Parser for vae'
parser = ArgumentParser(description=usage)
parser.add_argument(
    '--task_name', type=str, default='vae',
                    help='seed for np')
parser.add_argument(
    '--dataset_name', type=str, default='mnist',
    help='Which Dataset to train on, out of I128, I256, C10, C100; (default: %(default)s)')
parser.add_argument(
    '--tfds_dir', type=str, default='/gdata/tfds',
    help='Default location where data is stored (default: %(default)s)')
args = vars(parser.parse_args())

tf.config.optimizer.set_jit(True)
