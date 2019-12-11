import tensorflow as tf
import numpy as np
from utils.utils import *
from datasets.datasets import get_dataset
import network.vae as vae
from argparse import ArgumentParser


EPS = 1e-10


def prepare_parser():
    usage = 'Parser for vae'
    parser = ArgumentParser(description=usage)
    parser.add_argument('--task_name', type=str, default='vae',
                        help='seed for np')
    parser.add_argument(
        '--dataset_name', type=str, default='mnist',
        help='Which Dataset to train on, out of I128, I256, C10, C100; (default: %(default)s)')
    parser.add_argument(
        '--tfds_dir', type=str, default='/gdata/tfds',
        help='Default location where data is stored (default: %(default)s)')
    parser.add_argument(
        '--batch_size', type=int, default=256,
        help='Default overall batchsize (default: %(default)s)')
    parser.add_argument(
        '--gpu_num', type=int, default=1,
        help='Number of dataloader workers (default: %(default)s)')
    parser.add_argument(
        '--dim_z', type=int, default=10,
        help='Default overall batchsize (default: %(default)s)')
    parser.add_argument(
        '--e_hidden_num', type=int, default=10,
        help='Default overall batchsize (default: %(default)s)')
    parser.add_argument(
        '--d_hidden_num', type=int, default=300,
        help='Default overall batchsize (default: %(default)s)')
    parser.add_argument(
        '--lr', type=float, default=0.0001,
        help='Default overall batchsize (default: %(default)s)')
    parser.add_argument(
        '--decay_step', type=int, default=3000,
        help='Default overall batchsize (default: %(default)s)')
    parser.add_argument(
        '--decay_coef', type=int, default=0.5,
        help='Default overall batchsize (default: %(default)s)')
    parser.add_argument(
        '--beta2', type=int, default=0.999,
        help='Default overall batchsize (default: %(default)s)')
    parser.add_argument(
        '--example_nums', type=int, default=32,
        help='Default overall batchsize (default: %(default)s)')
    parser.add_argument(
        '--num_midpoints', type=int, default=16,
        help='Default overall batchsize (default: %(default)s)')
    parser.add_argument(
        '--resume', action='store_true', default=False,
        help='seed for np')
    parser.add_argument(
        '--restore_e_dir', type=str, default='/ghome/fengrl/disc_ckpt/disc-0',
        help='seed for np')
    parser.add_argument(
        '--restore_d_dir', type=str, default='/ghome/fengrl/disc_ckpt/disc-0',
        help='seed for np')
    parser.add_argument(
        '--total_step', type=int, default=250000,
        help='seed for np')
    parser.add_argument(
        '--eval_per_steps', type=int, default=2000,
        help='seed for np')
    parser.add_argument(
        '--save_per_steps', type=int, default=2000,
        help='seed for np')
    parser.add_argument(
        '--print_loss_per_steps', type=int, default=2000,
        help='Use LZF compression? (default: %(default)s)')
    return parser


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
    Encoder = vae.Encoder(config.dim_z, config.e_hidden_num, exceptions=['opt'], name='Encoder')
    Decoder = vae.Decoder(config.img_shape, config.d_hidden_num, exceptions=['opt'], name='Decoder')
    learning_rate = tf.train.exponential_decay(config.lr, global_step, config.decay_step,
                                               config.decay_coef, staircase=False)
    solver = tf.train.AdamOptimizer(learning_rate=learning_rate, name='opt', beta2=config.beta2)
    print("Building tensorflow graph...")

    def train_step(image):
        mu_z, log_sigma_z, z = Encoder(image, is_training=True)
        x = Decoder(z, is_training=True, flatten=False)
        with tf.variable_scope('kl_divergence'):
            kl_divergence = - tf.reduce_mean(tf.reduce_sum(
                0.5 * (1 + log_sigma_z - mu_z ** 2 - tf.exp(log_sigma_z)), 1))
        with tf.variable_scope('reconstruction_loss'):
            recon_loss = - tf.reduce_mean(tf.reduce_sum(
                image * tf.log(x + EPS) + (1 - image) * tf.log(1 - x + EPS), [1, 2, 3]))
        loss = kl_divergence + recon_loss
        add_global = global_step.assign_add(1)
        update_ops = tf.get_collection(tf.GraphKeys.UPDATE_OPS)
        with tf.control_dependencies([add_global] + update_ops):
            opt = solver.minimize(loss, var_list=Encoder.trainable_variables + Decoder.trainable_variables)
            with tf.control_dependencies([opt]):
                return tf.identity(loss), tf.identity(recon_loss), \
                       tf.identity(kl_divergence)

    loss, r_loss, kl_loss = train_step(dataset.get_next()[0])
    print("Building eval module...")

    fixed_z = tf.constant(np.random.normal(size=[config.example_nums, config.dim_z]), dtype=tf.float32)
    fixed_z0 = tf.constant(np.random.normal(size=[config.example_nums, config.dim_z]), dtype=tf.float32)
    fixed_z1 = tf.constant(np.random.normal(size=[config.example_nums, config.dim_z]), dtype=tf.float32)
    fixed_x = tf.placeholder(tf.float32, [config.example_nums] + config.img_shape)
    fixed_x0 = tf.placeholder(tf.float32, [config.example_nums] + config.img_shape)
    fixed_x1 = tf.placeholder(tf.float32, [config.example_nums] + config.img_shape)
    input_dict = {'fixed_z': fixed_z, 'fixed_z0': fixed_z0, 'fixed_z1': fixed_z1, 'fixed_x': fixed_x,
                  'fixed_x0': fixed_x0, 'fixed_x1': fixed_x1, 'num_midpoints': config.num_midpoints}

    def eval_step():
        out_dict = generate_sample(Decoder, input_dict)
        out_dict.update(reconstruction_sample(Encoder, Decoder, input_dict))
        out_dict.update({'fixed_x': fixed_x, 'fixed_x0': fixed_x0, 'fixed_x1': fixed_x1})
        return out_dict

    o_dict = eval_step()

    print("Building init module...")
    with tf.init_scope():
        init = [tf.global_variables_initializer(), dataset.initializer, eval_dataset.initializer]
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
        print('Preparing eval utils...')

        fixed_x_, _ = get_fixed_x(sess, dataset, config.example_nums, config.batch_size)
        fixed_x0_, _ = get_fixed_x(sess, dataset, config.example_nums, config.batch_size)
        fixed_x1_, _ = get_fixed_x(sess, dataset, config.example_nums, config.batch_size)
        print("Completing all work, iteration now start, consuming %s " % timer.runing_time_format)
        print("Start iterations...")
        for iteration in range(config.total_step):
            loss_, r_loss_, kl_loss_, lr_ = sess.run([loss, r_loss, kl_loss, learning_rate])
            if iteration % config.print_loss_per_steps == 0:
                timer.update()
                print("step %d, loss %f, r_loss_ %f, kl_loss_ %f, learning_rate % f, consuming time %s" %
                      (iteration, loss_, r_loss_, kl_loss_, lr_, timer.runing_time_format))
            if iteration % config.eval_per_steps == 0:
                o_dict_ = sess.run(o_dict, {fixed_x: fixed_x_, fixed_x0: fixed_x0_, fixed_x1: fixed_x1_})
                for key in o_dict:
                    if not os.path.exists(config.model_dir + '/%06d' % iteration):
                        os.makedirs(config.model_dir + '/%06d' % iteration)
                    save_image_grid(o_dict_[key], config.model_dir + '/%06d/%s.jpg' % (iteration, key))
            if iteration % config.save_per_steps == 0:
                saver_e.save(sess, save_path=config.model_dir + '/en.ckpt',
                             global_step=iteration, write_meta_graph=False)
                saver_d.save(sess, save_path=config.model_dir + '/de.ckpt',
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



