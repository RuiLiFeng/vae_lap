import tensorflow as tf
import numpy as np
from utils.utils import *
from datasets.datasets import get_dataset
import network.vae_celeba as vae
import functools


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
    global_step = tf.get_variable(name='global_step', initializer=tf.constant(0), trainable=False,
                                  aggregation=tf.VariableAggregation.ONLY_FIRST_REPLICA)
    print("Constructing networks...")
    if config.laplace_lambda != 0:
        Smoother = vae.Encoder(config.dim_z, exceptions=['opt'], name='VAE_En')

    Encoder = vae.Encoder(config.dim_z, exceptions=['opt'], name='Encoder')
    Decoder = vae.Decoder(dset.image_shape, exceptions=['opt'], name='Decoder')
    learning_rate = tf.train.exponential_decay(config.lr, global_step, config.decay_step,
                                               config.decay_coef, staircase=False)
    solver = tf.train.AdamOptimizer(learning_rate=learning_rate, name='opt', beta2=config.beta2)
    print("Building tensorflow graph...")

    def train_step(image):
        mu_z, log_sigma_z, z = Encoder(image, is_training=True)
        x = Decoder(z, is_training=True)
        with tf.variable_scope('kl_divergence'):
            kl_divergence = - tf.reduce_mean(tf.reduce_sum(
                0.5 * (1 + log_sigma_z - mu_z ** 2 - tf.exp(log_sigma_z)), 1))
        with tf.variable_scope('reconstruction_loss'):
            recon_loss = config.sigma ** 2 * tf.reduce_mean(tf.reduce_sum(
                tf.square(image - x), [1, 2, 3]))
        if config.laplace_lambda != 0:
            _, _, y = Smoother(image, True)
            with tf.variable_scope('smooth_loss'):
                s_w = smoother_weight(y, 'heat', sigma=config.smooth_sigma)
                smooth_loss = batch_laplacian(s_w, z) * config.laplace_lambda
        else:
            smooth_loss = 0.0
        if config.laplace_lambda_x != 0:
            with tf.variable_scope('smooth_loss_x'):
                sx_w = smoother_weight(z, 'heat', sigma=config.smooth_sigma_x)
                smooth_loss_x = batch_laplacian(sx_w, tf.reshape(x, [x.shape[0], -1])) * config.laplace_lambda_x
        else:
            smooth_loss_x = 0.0
        loss = kl_divergence + recon_loss + smooth_loss + smooth_loss_x
        add_global = global_step.assign_add(1)
        update_ops = tf.get_collection(tf.GraphKeys.UPDATE_OPS)
        with tf.control_dependencies([add_global] + update_ops):
            opt = solver.minimize(loss, var_list=Encoder.trainable_variables + Decoder.trainable_variables)
            with tf.control_dependencies([opt]):
                return tf.identity(loss), tf.identity(recon_loss), \
                       tf.identity(kl_divergence), tf.identity(smooth_loss), tf.identity(smooth_loss_x)

    loss, r_loss, kl_loss, s_loss, sx_loss = train_step(dataset.get_next()[0])
    print("Building eval module...")

    fixed_z = tf.constant(np.random.normal(size=[config.example_nums, config.dim_z]), dtype=tf.float32)
    fixed_z0 = tf.constant(np.random.normal(size=[config.example_nums, config.dim_z]), dtype=tf.float32)
    fixed_z1 = tf.constant(np.random.normal(size=[config.example_nums, config.dim_z]), dtype=tf.float32)
    fixed_x = tf.placeholder(tf.float32, (config.example_nums,) + dset.image_shape)
    fixed_x0 = tf.placeholder(tf.float32, (config.example_nums,) + dset.image_shape)
    fixed_x1 = tf.placeholder(tf.float32, (config.example_nums,) + dset.image_shape)
    input_dict = {'fixed_z': fixed_z, 'fixed_z0': fixed_z0, 'fixed_z1': fixed_z1, 'fixed_x': fixed_x,
                  'fixed_x0': fixed_x0, 'fixed_x1': fixed_x1, 'num_midpoints': config.num_midpoints}

    def sample_step():
        out_dict = generate_sample(Decoder, input_dict)
        out_dict.update(reconstruction_sample(Encoder, Decoder, input_dict))
        out_dict.update({'fixed_x': fixed_x, 'fixed_x0': fixed_x0, 'fixed_x1': fixed_x1})
        return out_dict

    def eval_step(image):
        mu_z, log_sigma_z, z = Encoder(image, is_training=True)
        x = Decoder(z, is_training=True)
        mse = tf.reduce_mean(tf.reduce_sum(tf.square(image - x), axis=[1, 2, 3]))
        return mse

    mse = eval_step(dataset.get_next()[0])

    o_dict = sample_step()

    print("Building init module...")
    with tf.init_scope():
        init = [tf.global_variables_initializer(), dataset.initializer, eval_dataset.initializer]
        saver_e = tf.train.Saver(Encoder.restore_variables)
        saver_d = tf.train.Saver(Decoder.restore_variables)
        if config.laplace_lambda != 0:
            saver_s = tf.train.Saver(Smoother.restore_variables)

    print('Starting training...')
    with tf.Session(config=tf.ConfigProto(allow_soft_placement=True)) as sess:
        sess.run(init)
        if config.laplace_lambda != 0:
            print('Restore smoother...')
            saver_s.restore(sess, config.restore_s_dir)
        if config.resume:
            print("Restore vae...")
            saver_e.restore(sess, config.restore_e_dir)
            saver_d.restore(sess, config.restore_d_dir)
        timer.update()
        print('Preparing sample utils...')

        fixed_x_, _ = get_fixed_x(sess, dataset, config.example_nums, config.batch_size)
        fixed_x0_, _ = get_fixed_x(sess, dataset, config.example_nums, config.batch_size)
        fixed_x1_, _ = get_fixed_x(sess, dataset, config.example_nums, config.batch_size)
        print("Completing all work, iteration now start, consuming %s " % timer.runing_time_format)
        print("Start iterations...")
        for iteration in range(config.total_step):
            loss_, r_loss_, kl_loss_, s_loss_, sx_loss_, lr_ = \
                sess.run([loss, r_loss, kl_loss, s_loss, sx_loss, learning_rate])
            if iteration % config.print_loss_per_steps == 0:
                mse_ = sess.run(mse)
                timer.update()
                print("step %d, loss %f, r_loss_ %f, kl_loss_ %f, s_loss %f, sx_loss %f, mse %f, "
                      "learning_rate % f, consuming time %s" %
                      (iteration, loss_, r_loss_, kl_loss_, s_loss_, sx_loss_, mse_,
                       lr_, timer.runing_time_format))
            if iteration % config.eval_per_steps == 0:
                o_dict_ = sess.run(o_dict, {fixed_x: fixed_x_, fixed_x0: fixed_x0_, fixed_x1: fixed_x1_})
                for key in o_dict:
                    if not os.path.exists(config.model_dir + '/%06d' % iteration):
                        os.makedirs(config.model_dir + '/%06d' % iteration)
                    if o_dict_[key].ndim == 5:
                        img = o_dict_[key].transpose([0, 1, 4, 2, 3])
                    else:
                        img = o_dict_[key].transpose([0, 3, 1, 2])
                    save_image_grid(img, config.model_dir + '/%06d/%s.jpg' % (iteration, key))
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


def batch_laplacian(weight, z):
    """

    :param weight:
    :param z: [bs, z_dim]
    :return:
    """
    z_x = tf.tile(tf.expand_dims(z, 1), [1, z.shape[0].value, 1])
    z_y = tf.tile(tf.expand_dims(z, 0), [z.shape[0].value, 1, 1])
    z_pairwise = tf.reduce_sum(tf.square(z_x - z_y), axis=2)
    return tf.reduce_mean(weight * z_pairwise)


def smoother_weight(y, kernel='heat', margin=None, sigma=2.5):
    y1_ = tf.tile(tf.expand_dims(y, 1), [1, y.shape[0].value, 1])
    y2_ = tf.tile(tf.expand_dims(y, 0), [y.shape[0].value, 1, 1])
    pairwise_dis = tf.reduce_sum(tf.square(y1_ - y2_), axis=2)
    if kernel == 'heat':
        w = tf.exp(-pairwise_dis / sigma ** 2)
    else:
        raise ValueError('Unsupported kernel type! %s' % kernel)
    return w
