import tensorflow as tf
import numpy as np
from utils.utils import *
from datasets.datasets import get_dataset
import network.vae as vae
import network.arch_ops as ops
from argparse import ArgumentParser


EPS = 1e-10


def training_loop(config: Config):
    timer = Timer()
    print('Task name %s' % config.task_name)
    print('Loading %s dataset...' % config.dataset_name)
    dset = get_dataset(config.dataset_name, config.tfds_dir, config.gpu_nums * 2)
    dataset = dset.input_fn(config.batch_size, mode='train')
    dataset = dataset.make_initializable_iterator()

    eval_dataset = dset.input_fn(config.batch_size, mode='eval')
    eval_dataset = eval_dataset.make_initializable_iterator()
    global_step = tf.get_variable(name='global_step', initializer=tf.constant(0), trainable=False)
    print("Constructing networks...")
    Encoder = vae.Encoder(config.dim_z, config.e_hidden_num, exceptions=['opt'], name='Classifier')
    last_layer = layer(config.dim_z)
    learning_rate = tf.train.exponential_decay(config.lr, global_step, config.decay_step,
                                               config.decay_coef, staircase=False)
    solver = tf.train.AdamOptimizer(learning_rate=learning_rate, name='opt', beta2=config.beta2)
    print("Building tensorflow graph...")

    def train_step(data):
        image, label = data
        mu_z, log_sigma_z, z = Encoder(image, is_training=True)
        label = tf.one_hot(label, 10)
        y = last_layer(z, True)
        loss = tf.nn.softmax_cross_entropy_with_logits(logits=y, labels=label)
        loss = tf.reduce_mean(loss)
        add_global = global_step.assign_add(1)
        update_ops = tf.get_collection(tf.GraphKeys.UPDATE_OPS)
        with tf.control_dependencies([add_global] + update_ops):
            opt = solver.minimize(loss, var_list=Encoder.trainable_variables + last_layer.trainable_variables)
            with tf.control_dependencies([opt]):
                return tf.identity(loss)

    loss = train_step(dataset.get_next())

    def eval_step(data):
        image, label = data
        mu_z, log_sigma_z, z = Encoder(image, is_training=True)
        y = last_layer(z, True)
        y = tf.nn.softmax(y)
        y = tf.arg_max(y, 1)
        p = tf.reduce_mean(tf.cast(tf.equal(y, label), tf.float32))
        return p

    p = eval_step(eval_dataset.get_next())

    print("Building init module...")
    with tf.init_scope():
        init = [tf.global_variables_initializer(), dataset.initializer, eval_dataset.initializer]
        saver_e = tf.train.Saver(Encoder.restore_variables)

    print('Starting training...')
    with tf.Session(config=tf.ConfigProto(allow_soft_placement=True)) as sess:
        sess.run(init)
        if config.resume:
            print("Restore vae...")
            saver_e.restore(sess, config.restore_e_dir)
        timer.update()
        print('Preparing eval utils...')

        fixed_x_, _ = get_fixed_x(sess, dataset, config.example_nums, config.batch_size)
        fixed_x0_, _ = get_fixed_x(sess, dataset, config.example_nums, config.batch_size)
        fixed_x1_, _ = get_fixed_x(sess, dataset, config.example_nums, config.batch_size)
        print("Completing all work, iteration now start, consuming %s " % timer.runing_time_format)
        print("Start iterations...")
        for iteration in range(config.total_step):
            loss_, lr_ = sess.run([loss, learning_rate])
            if iteration % config.print_loss_per_steps == 0:
                timer.update()
                print("step %d, loss %f, learning_rate % f, consuming time %s" %
                      (iteration, loss_, lr_, timer.runing_time_format))
            if iteration % 1000 == 0:
                p_ = 0.0
                for _ in range(10):
                    p_plus = sess.run(p)
                    p_ += p_plus
                p_ /= 10
                print('precise in eval %f' % p_)
            if iteration % config.save_per_steps == 0:
                saver_e.save(sess, save_path=config.model_dir + '/en.ckpt',
                             global_step=iteration, write_meta_graph=False)


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


class layer(object):
    def __init__(self, dim_z):
        self.dim_z = dim_z
        self.name = 'last_layer'

    def __call__(self, z, is_training, reuse=tf.AUTO_REUSE):
        with tf.variable_scope(self.name, values=[z], reuse=reuse):
            y = ops.linear(z, 10, 'last_fn')
        return y

    @property
    def trainable_variables(self):
        return [var for var in tf.trainable_variables() if self.name in var.name]
