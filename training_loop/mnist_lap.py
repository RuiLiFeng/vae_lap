import tensorflow as tf
import numpy as np
from utils.utils import *
from datasets.datasets import get_dataset, load_mnist_from_record
import network.vae as vae
import network.arch_ops as ops
from argparse import ArgumentParser


EPS = 1e-10


def training_loop(config: Config):
    timer = Timer()
    print('Task name %s' % config.task_name)
    print('Loading %s dataset...' % config.dataset_name)
    dataset = load_mnist_from_record(config.record_dir + '/Mnist20_rep.tfrecords', config.batch_size)
    dataset = dataset.make_initializable_iterator()
    laplace_sigma2 = np.load(config.record_dir + '/sigma2.npy') / (-np.log(config.laplace_a))

    global_step = tf.get_variable(name='global_step', initializer=tf.constant(0), trainable=False)
    print("Constructing networks...")
    Encoder = vae.Encoder(config.dim_z, config.e_hidden_num, exceptions=['opt'], name='Encoder')
    Decoder = vae.Decoder(config.img_shape, config.d_hidden_num, exceptions=['opt'], name='Decoder')
    learning_rate = tf.train.exponential_decay(config.lr, global_step, config.decay_step,
                                               config.decay_coef, staircase=False)
    solver = tf.train.AdamOptimizer(learning_rate=learning_rate, name='opt', beta2=config.beta2)
    print("Building tensorflow graph...")

    def train_step(data):
        image, rep, label = data
        mu_z, log_sigma_z, z = Encoder(image, is_training=True)
        x = Decoder(z, is_training=True, flatten=False)
        with tf.variable_scope('kl_divergence'):
            kl_divergence = - tf.reduce_mean(tf.reduce_sum(
                0.5 * (1 + log_sigma_z - mu_z ** 2 - tf.exp(log_sigma_z)), 1))
        with tf.variable_scope('reconstruction_loss'):
            recon_loss = - tf.reduce_mean(tf.reduce_sum(
                image * tf.log(x + EPS) + (1 - image) * tf.log(1 - x + EPS), [1, 2, 3]))
        with tf.variable_scope('smooth_loss'):
            s_w = smoother_weight(rep, 'heat', sigma2=laplace_sigma2)
            smooth_loss = batch_laplacian(s_w, z) * config.laplace_lambda
        loss = kl_divergence + recon_loss + smooth_loss
        add_global = global_step.assign_add(1)
        update_ops = tf.get_collection(tf.GraphKeys.UPDATE_OPS)
        with tf.control_dependencies([add_global] + update_ops):
            opt = solver.minimize(loss, var_list=Encoder.trainable_variables + Decoder.trainable_variables)
            with tf.control_dependencies([opt]):
                return tf.identity(loss), tf.identity(recon_loss), \
                       tf.identity(kl_divergence), tf.identity(smooth_loss), tf.identity(s_w)

    loss, r_loss, kl_loss, s_loss, s_w = train_step(dataset.get_next())
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
        print('Preparing eval utils...')

        fixed_x_, _ = get_fixed_x(sess, dataset, config.example_nums, config.batch_size)
        fixed_x0_, _ = get_fixed_x(sess, dataset, config.example_nums, config.batch_size)
        fixed_x1_, _ = get_fixed_x(sess, dataset, config.example_nums, config.batch_size)
        print("Completing all work, iteration now start, consuming %s " % timer.runing_time_format)
        print("Start iterations...")
        for iteration in range(config.total_step):
            loss_, r_loss_, kl_loss_, s_loss_, sw_sum_, lr_ = \
                sess.run([loss, r_loss, kl_loss, s_loss, s_w, learning_rate])
            if iteration % config.print_loss_per_steps == 0:
                timer.update()
                print("step %d, loss %f, r_loss_ %f, kl_loss_ %f, s_loss_ %f, sw_prod %f, "
                      "learning_rate % f, consuming time %s" %
                      (iteration, loss_, r_loss_, kl_loss_, s_loss_, np.prod(sw_sum_)**(1/255**2),
                       lr_, timer.runing_time_format))
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


def smoother_weight(y, kernel='heat', margin=0.4, sigma2=2.5):
    y1_ = tf.tile(tf.expand_dims(y, 1), [1, y.shape[0].value, 1])
    y2_ = tf.tile(tf.expand_dims(y, 0), [y.shape[0].value, 1, 1])
    pairwise_dis = tf.reduce_sum(tf.square(y1_ - y2_), axis=2)
    if kernel == 'heat':
        dis = pairwise_dis * 256 / sigma2
        w = tf.exp(-dis)
        w = tf.nn.relu(ops.standardize_batch(w, True)) / 2 + 0.5
        # w = w * tf.cast(w > margin, tf.float32)

    else:
        raise ValueError('Unsupported kernel type! %s' % kernel)
    return w
