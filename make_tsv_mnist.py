import tensorflow as tf
import numpy as np
import network.vae as nn
from datasets.datasets import get_dataset
import csv
import os
from argparse import ArgumentParser
from tqdm import tqdm

usage = 'Parser for vae'
parser = ArgumentParser(description=usage)
parser.add_argument(
    '--dataset_name', type=str, default='mnist',
    help='Which Dataset to train on, out of I128, I256, C10, C100; (default: %(default)s)')
parser.add_argument(
    '--batch_size', type=int, default=256,
    help='Which Dataset to train on, out of I128, I256, C10, C100; (default: %(default)s)')
parser.add_argument(
    '--total_step', type=int, default=200,
    help='Which Dataset to train on, out of I128, I256, C10, C100; (default: %(default)s)')
parser.add_argument(
    '--dim_z', type=int, default=20,
    help='Which Dataset to train on, out of I128, I256, C10, C100; (default: %(default)s)')
parser.add_argument(
    '--e_hidden_num', type=int, default=100,
    help='Which Dataset to train on, out of I128, I256, C10, C100; (default: %(default)s)')
parser.add_argument(
    '--model_dir', type=str, default='/gdata/vsiual/mnist20',
    help='Which Dataset to train on, out of I128, I256, C10, C100; (default: %(default)s)')


def main(config):
    print('Loading %s dataset...' % config['dataset_name'])
    dset = get_dataset(config['dataset_name'], '/gdata/tfds', 2)
    dataset = dset.input_fn(config['batch_size'], mode='train')
    dataset = dataset.make_initializable_iterator()
    Encoder = nn.Encoder(config['dim_z'], config['e_hidden_num'], exceptions=['opt'], name='VAE_En')
    image, label = dataset.get_next()
    _, _, z = Encoder(image, is_training=True)
    saver = tf.train.Saver(Encoder.restore_variables)
    with tf.Session(config=tf.ConfigProto(allow_soft_placement=True)) as sess:
        sess.run([tf.global_variables_initializer(), dataset.initializer])
        print("Restore Encoder...")
        saver.restore(sess, config['model_dir'] + '/en.ckpt-62000')
        print('Generate embeddings...')

        f = open(config['model_dir'] + '/embeddings.tsv', 'wt')
        f_writer = csv.writer(f, delimiter='\t')
        g = open(config['model_dir'] + '/labels.tsv', 'wt')
        g_writer = csv.writer(g, delimiter='\t')

        for _ in tqdm(range(config['total_step'])):
            z_, l_ = sess.run([z, label])
            for row in z_:
                f_writer.writerow(row)
            for row in l_:
                g_writer.writerow([row])
        f.close()
        g.close()


args = vars(parser.parse_args())
print(args)
main(args)
