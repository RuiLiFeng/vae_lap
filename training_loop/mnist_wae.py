import tensorflow as tf
import numpy as np
from utils.utils import *
from datasets.datasets import get_dataset, load_mnist_KNN_from_record
import network.wvae as vae
import network.arch_ops as ops
import utils.wae_config as w_config
from argparse import ArgumentParser


EPS = 1e-10


def training_loop(config: Config):
    timer = Timer()
    opts = w_config.config_mnist
    print('Task name %s' % config.task_name)
    print('Loading %s dataset...' % config.dataset_name)
    dataset = load_mnist_KNN_from_record(config.record_dir + '/Mnist20knn5_rep.tfrecords', config.batch_size)
    dataset = dataset.make_initializable_iterator()
    laplace_sigma2 = np.load(config.record_dir + '/knn5sigma2.npy') / (-np.log(config.laplace_a))

    global_step = tf.get_variable(name='global_step', initializer=tf.constant(0), trainable=False)
    print("Constructing networks...")
    Encoder = vae.Encoder(opts, exceptions=['opt'], name='Encoder')
    Decoder = vae.Decoder(opts, exceptions=['opt'], name='Decoder')
    Discriminator = vae.Discriminator(opts, exceptions=['opt'])
    learning_rate = tf.train.exponential_decay(opts['lr'], global_step, config.decay_step,
                                               config.decay_coef, staircase=False)
    solver = tf.train.AdamOptimizer(learning_rate=learning_rate, name='opt', beta1=opts['adam_beta1'])
    adv_solver = tf.train.AdamOptimizer(learning_rate=learning_rate, name='opt', beta1=opts['adam_beta1'])

    print("Building tensorflow graph...")

    def train_step(data):
        image, rep, label, neighbour, index = data
        mu_z, log_sigma_z, z = Encoder(image, is_training=True)
        x = Decoder(z, is_training=True, flatten=False)
        # with tf.variable_scope('kl_divergence'):
        #     kl_divergence = - tf.reduce_mean(tf.reduce_sum(
        #         0.5 * (1 + log_sigma_z - mu_z ** 2 - tf.exp(log_sigma_z)), 1))
        with tf.variable_scope('reconstruction_loss'):
            # recon_loss = - tf.reduce_mean(tf.reduce_sum(
            #     image * tf.log(x + EPS) + (1 - image) * tf.log(1 - x + EPS), [1, 2, 3]))
            recon_loss = 0.05 * tf.reduce_mean(tf.reduce_sum(tf.square(image - x), [1, 2, 3]))
        # with tf.variable_scope('smooth_loss'):
        #     mask = make_mask(neighbour, index)
        #     s_w = mask * smoother_weight(rep, 'heat', sigma2=laplace_sigma2)
        #     smooth_loss = batch_laplacian(s_w, z) * config.laplace_lambda
        with tf.variable_scope('wae_penalty'):
            Pz = tf.random.normal(shape=[config.batch_size, config.dim_z], mean=0.0, stddev=1.0)
            logits_Pz = Discriminator(Pz, True)
            logits_Qz = Discriminator(z, True)
            loss_Pz = tf.reduce_mean(
                tf.nn.sigmoid_cross_entropy_with_logits(
                    logits=logits_Pz, labels=tf.ones_like(logits_Pz)))
            loss_Qz = tf.reduce_mean(
                tf.nn.sigmoid_cross_entropy_with_logits(
                    logits=logits_Qz, labels=tf.zeros_like(logits_Qz)))
            loss_Qz_trick = tf.reduce_mean(
                tf.nn.sigmoid_cross_entropy_with_logits(
                    logits=logits_Qz, labels=tf.ones_like(logits_Qz)))
            loss_adv = config.wae_lambda * (loss_Pz + loss_Qz)
            loss_match = config.wae_lambda * loss_Qz_trick
        # loss = kl_divergence + recon_loss + smooth_loss
        loss = loss_match + recon_loss
        add_global = global_step.assign_add(1)
        update_ops = tf.get_collection(tf.GraphKeys.UPDATE_OPS)
        with tf.control_dependencies([add_global] + update_ops):
            opt = solver.minimize(loss, var_list=Encoder.trainable_variables + Decoder.trainable_variables)
            with tf.control_dependencies([opt]):
                l1, l2, l3 = tf.identity(loss), tf.identity(recon_loss), \
                       tf.identity(loss_match)
        with tf.control_dependencies([add_global] + update_ops):
            d_opt = adv_solver.minimize(loss_adv, var_list=Discriminator.trainable_variables)
            with tf.control_dependencies([d_opt]):
                l6 = tf.identity(loss_adv)
        return l1, l2, l3, l6

    def pretrain(data):
        image, rep, label, neighbour, index = data
        mu_z, log_sigma_z, z = Encoder(image, is_training=True)
        Pz = tf.random.normal(shape=[config.batch_size, config.dim_z], mean=0.0, stddev=1.0)
        mean_pz = tf.reduce_mean(Pz, axis=0, keep_dims=True)
        mean_qz = tf.reduce_mean(z, axis=0, keep_dims=True)
        mean_loss = tf.reduce_mean(tf.square(mean_pz - mean_qz))
        cov_pz = tf.matmul(Pz - mean_pz, Pz - mean_pz,
                           transpose_a=True) / (config.batch_size - 1)
        cov_qz = tf.matmul(z - mean_qz, z - mean_qz,
                           transpose_a=True) / (config.batch_size - 1)
        cov_loss = tf.reduce_mean(tf.square(cov_pz - cov_qz))
        pretrain_loss = cov_loss + mean_loss
        update_ops = tf.get_collection(tf.GraphKeys.UPDATE_OPS)
        with tf.control_dependencies(update_ops):
            opt = solver.minimize(pretrain_loss, var_list=Encoder.trainable_variables)
            with tf.control_dependencies([opt]):
                p_loss = tf.identity(pretrain_loss)
        return p_loss

    p_loss = pretrain(dataset.get_next())
    loss, r_loss, m_loss, a_loss = train_step(dataset.get_next())
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
        print("Start pretraining of Encoder...")
        for iteration in range(500):
            p_loss_ = sess.run(p_loss)
            if iteration % 50 == 0:
                print("Pretrain_step %d, p_loss %f" % (iteration, p_loss_))
        print("Pretraining of Encoder Done! p_loss %f. Now start training..." % p_loss_)
        for iteration in range(config.total_step):
            loss_, r_loss_, m_loss_, lr_ = \
                sess.run([loss, r_loss, m_loss, learning_rate])
            a_loss_ = sess.run(a_loss)

            if iteration % config.print_loss_per_steps == 0:
                timer.update()
                print("step %d, loss %f, r_loss_ %f, m_loss_ %f, a_loss %f "
                      "learning_rate % f, consuming time %s" %
                      (iteration, loss_, r_loss_, m_loss_, a_loss_,
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
        dis = pairwise_dis / sigma2
        w = tf.exp(-dis)
        w = tf.nn.relu(ops.standardize_batch(w, True)) / 2 + 0.5
        # w = w * tf.cast(w > margin, tf.float32)

    else:
        raise ValueError('Unsupported kernel type! %s' % kernel)
    return w


def make_mask(neighbour, index):
    index = tf.tile(tf.expand_dims(tf.expand_dims(index, 1), 0), [index.shape[0].value, 1, 50])
    neighbour = tf.tile(tf.expand_dims(neighbour, 1), [1, index.shape[0].value, 1])
    mask = tf.reduce_sum(tf.cast(tf.equal(index, neighbour), tf.int32), 2)
    mask = tf.cast(tf.greater(mask + tf.transpose(mask), 0), tf.float32)
    return mask


