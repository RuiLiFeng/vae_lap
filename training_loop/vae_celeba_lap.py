import tensorflow as tf
import numpy as np
from utils.utils import *
from datasets.datasets import get_dataset, load_CelebA_KNN_from_record
import network.vae_celeba as vae
import functools
from metric import perceptual_path_length as ppl


EPS = 1e-10


def training_loop(config: Config):
    timer = Timer()
    print('Task name %s' % config.task_name)
    print('Loading %s dataset...' % config.dataset_name)
    dataset = load_CelebA_KNN_from_record(config.record_dir + '/CelebA64knn5_rep.tfrecords', config.batch_size)
    dataset = dataset.make_initializable_iterator()
    laplace_sigma2 = 1.0 / (-np.log(config.laplace_a))
    global_step = tf.get_variable(name='global_step', initializer=tf.constant(0), trainable=False,
                                  aggregation=tf.VariableAggregation.ONLY_FIRST_REPLICA)
    print("Constructing networks...")

    Encoder = vae.Encoder(config.dim_z, exceptions=['opt'], name='VAE_En')
    Decoder = vae.Decoder([64, 64, 3], exceptions=['opt'], name='VAE_De')
    valina_encoder = vae.Encoder(config.dim_z, exceptions=['opt'], name='Encoder')

    def lip_metric(inputs):
        return inputs

    def d_metric(inputs):
        _, _, outputs = valina_encoder(inputs, True)
        return outputs

    def generator(inputs):
        outputs = Decoder(inputs, True, False)
        return outputs

    def lip_generator(inputs):
        _, _, outputs = Encoder(inputs, True)
        return outputs

    PPL = ppl.PPL_mnist(epsilon=0.01, sampling='full', generator=generator, d_metric=d_metric)
    Lip_PPL = ppl.PPL_mnist(epsilon=0.01, sampling='full', generator=lip_generator, d_metric=lip_metric)
    learning_rate = tf.train.exponential_decay(config.lr, global_step, config.decay_step,
                                               config.decay_coef, staircase=False)
    solver = tf.train.AdamOptimizer(learning_rate=learning_rate, name='opt', beta2=config.beta2)
    print("Building tensorflow graph...")

    def train_step(data):
        image, rep, label, neighbour, index = data
        mu_z, log_sigma_z, z = Encoder(image, is_training=True)
        x = Decoder(z, is_training=True, flatten=False)
        with tf.variable_scope('kl_divergence'):
            kl_divergence = - config.wae_lambda * tf.reduce_mean(tf.reduce_sum(
                0.5 * (1 + log_sigma_z - mu_z ** 2 - tf.exp(log_sigma_z)), 1))
        with tf.variable_scope('reconstruction_loss'):
            recon_loss = config.sigma ** 2 * tf.reduce_mean(tf.reduce_sum(
                tf.square(image - x), [1, 2, 3]))
            # recon_loss = 0.05 * tf.reduce_mean(tf.reduce_sum(tf.square(image - x), [1, 2, 3]))
        with tf.variable_scope('smooth_loss'):
            mask = make_mask(neighbour, index)
            s_w = mask * smoother_weight(rep, 'heat', sigma2=laplace_sigma2, mask=mask)
            smooth_loss = batch_laplacian(s_w, z) * config.laplace_lambda
            s_w_mean = tf.reduce_mean(s_w) * config.batch_size * config.batch_size / (tf.reduce_sum(mask) + EPS)

        loss = kl_divergence + recon_loss + smooth_loss
        # loss = loss_match + recon_loss
        add_global = global_step.assign_add(1)
        update_ops = tf.get_collection(tf.GraphKeys.UPDATE_OPS)
        with tf.control_dependencies([add_global] + update_ops):
            opt = solver.minimize(loss, var_list=Encoder.trainable_variables + Decoder.trainable_variables)
            with tf.control_dependencies([opt]):
                l1, l2, l3, l4, l5 = tf.identity(loss), tf.identity(recon_loss), \
                                     tf.identity(kl_divergence), tf.identity(smooth_loss), tf.identity(s_w_mean)

        return l1, l2, l3, l4, l5

    loss, r_loss, kl_loss, s_loss, s_w = train_step(dataset.get_next())
    print("Building eval module...")

    fixed_z = tf.constant(np.random.normal(size=[config.example_nums, config.dim_z]), dtype=tf.float32)
    fixed_z0 = tf.constant(np.random.normal(size=[config.example_nums, config.dim_z]), dtype=tf.float32)
    fixed_z1 = tf.constant(np.random.normal(size=[config.example_nums, config.dim_z]), dtype=tf.float32)
    fixed_x = tf.placeholder(tf.float32, (config.example_nums,) + (64, 64, 3))
    fixed_x0 = tf.placeholder(tf.float32, (config.example_nums,) + (64, 64, 3))
    fixed_x1 = tf.placeholder(tf.float32, (config.example_nums,) + (64, 64, 3))
    input_dict = {'fixed_z': fixed_z, 'fixed_z0': fixed_z0, 'fixed_z1': fixed_z1, 'fixed_x': fixed_x,
                  'fixed_x0': fixed_x0, 'fixed_x1': fixed_x1, 'num_midpoints': config.num_midpoints}

    def sample_step():
        out_dict = generate_sample(Decoder, input_dict)
        out_dict.update(reconstruction_sample(Encoder, Decoder, input_dict))
        out_dict.update({'fixed_x': fixed_x, 'fixed_x0': fixed_x0, 'fixed_x1': fixed_x1})
        return out_dict

    o_dict = sample_step()

    def eval_step(img1, img2):
        z0 = tf.random.normal(shape=[config.batch_size, config.dim_z], mean=0.0, stddev=1.0)
        z1 = tf.random.normal(shape=[config.batch_size, config.dim_z], mean=0.0, stddev=1.0)
        _, _, img1_z = Encoder(img1, True)
        _, _, img2_z = Encoder(img2, True)
        ppl_sample_loss = PPL(z0, z1)
        ppl_de_loss = PPL(img1_z, img2_z)
        lip_loss = Lip_PPL(img1, img2)
        return ppl_sample_loss, ppl_de_loss, lip_loss

    img_1, _, _, _, _ = dataset.get_next()
    img_2, _, _, _, _ = dataset.get_next()
    ppl_sa_loss, ppl_de_loss, lip_loss = eval_step(img_1, img_2)

    print("Building init module...")
    with tf.init_scope():
        init = [tf.global_variables_initializer(), dataset.initializer]
        saver_e = tf.train.Saver(Encoder.restore_variables, max_to_keep=10)
        saver_d = tf.train.Saver(Decoder.restore_variables, max_to_keep=10)
        saver_v = tf.train.Saver(valina_encoder.restore_variables)

    print('Starting training...')
    with tf.Session(config=tf.ConfigProto(allow_soft_placement=True)) as sess:
        sess.run(init)
        saver_v.restore(sess, config.restore_s_dir)
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
        loss_list = []
        r_loss_list = []
        kl_loss_list = []
        s_loss_list = []
        ppl_sa_list = []
        ppl_re_list = []
        lip_list = []
        print("Start iterations...")
        for iteration in range(config.total_step):
            loss_, r_loss_, m_loss_, s_loss_, sw_sum_, lr_ = \
                sess.run([loss, r_loss, kl_loss, s_loss, s_w, learning_rate])
            if iteration % config.print_loss_per_steps == 0:
                loss_list.append(loss_)
                r_loss_list.append(r_loss_)
                kl_loss_list.append(m_loss_)
                s_loss_list.append(s_loss_)
                timer.update()
                print("step %d, loss %f, r_loss_ %f, kl_loss_ %f, s_loss_ %f, sw %f, "
                      "learning_rate % f, consuming time %s" %
                      (iteration, loss_, r_loss_, m_loss_, s_loss_, np.mean(sw_sum_),
                       lr_, timer.runing_time_format))
            if iteration % 1000 == 0:
                sa_loss_ = 0.0
                de_loss_ = 0.0
                lip_loss_ = 0.0
                for _ in range(200):
                    sa_p, de_p, lip_p = sess.run([ppl_sa_loss, ppl_de_loss, lip_loss])
                    sa_loss_ += sa_p
                    de_loss_ += de_p
                    lip_loss_ += lip_p
                sa_loss_ /= config.batch_size * 256
                de_loss_ /= config.batch_size * 256
                lip_loss_ /= config.batch_size * 256
                ppl_re_list.append(de_loss_)
                ppl_sa_list.append(sa_loss_)
                lip_list.append(lip_loss_)
                print("ppl_sample %f, ppl_resample %f, lipschitze %f" % (sa_loss_, de_loss_, lip_loss_))
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
                metric_dict = {'r': r_loss_list, 'm': kl_loss_list, 's': s_loss_list,
                               'psa': ppl_sa_list, 'pre': ppl_re_list, 'lip': lip_list}
                np.save(config.model_dir + '/%06d' % iteration + 'metric.npy', metric_dict)


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


def smoother_weight(y, kernel='heat', margin=0.4, sigma2=2.5, mask=None):
    y1_ = tf.tile(tf.expand_dims(y, 1), [1, y.shape[0].value, 1])
    y2_ = tf.tile(tf.expand_dims(y, 0), [y.shape[0].value, 1, 1])
    pairwise_dis = tf.reduce_sum(tf.square(y1_ - y2_), axis=2)
    norm = tf.expand_dims(EPS + tf.reduce_sum(pairwise_dis * mask, 1), 1)
    if kernel == 'heat':
        dis = pairwise_dis * tf.expand_dims(tf.reduce_sum(mask, 1), 1) / (sigma2 * norm)
        w = tf.exp(-dis)
        # w = tf.nn.relu(ops.standardize_batch(w, True)) / 2 + 0.5
        # w = w * tf.cast(w > margin, tf.float32)

    else:
        raise ValueError('Unsupported kernel type! %s' % kernel)
    return w


def make_mask(neighbour, index):
    index = tf.tile(tf.expand_dims(tf.expand_dims(index, 1), 0), [index.shape[0].value, 1, 55])
    neighbour = tf.tile(tf.expand_dims(neighbour, 1), [1, index.shape[0].value, 1])
    mask = tf.reduce_sum(tf.cast(tf.equal(index, neighbour), tf.int32), 2)
    mask = tf.cast(tf.greater(mask + tf.transpose(mask), 0), tf.float32)
    return mask


