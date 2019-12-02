import tensorflow as tf
import numpy as np
from utils.utils import *
from datasets.datasets import get_dataset
import network.vae as vae


EPS = 1e-10


def training_loop(config: Config):
    timer = Timer()
    print('Task name %s' % config.task_name)
    strategy = tf.distribute.MirroredStrategy()
    print('Loading %s dataset...' % config.dataset_name)
    dset = get_dataset(config.dataset_name, config.tfds_dir, config.gpu_nums * 2)
    dataset = dset.input_fn(config.batch_size, mode='train')
    dataset = strategy.experimental_distribute_dataset(dataset)
    dataset = dataset.make_initializable_iterator()

    eval_dataset = dset.input_fn(config.batch_size, mode='eval')
    eval_dataset = strategy.experimental_distribute_dataset(eval_dataset)
    eval_dataset = eval_dataset.make_initializable_iterator()
    with strategy.scope():
        global_step = tf.get_variable(name='global_step', initializer=tf.constant(0), trainable=False,
                                      aggregation=tf.VariableAggregation.ONLY_FIRST_REPLICA)
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
                kl_divergence = 0.5 * (1 + log_sigma_z - mu_z ** 2 - tf.exp(log_sigma_z))
            with tf.variable_scope('reconstruction_loss'):
                recon_loss = tf.reduce_mean(tf.nn.sigmoid_cross_entropy_with_logits(labels=image, logits=x))
            loss = kl_divergence + recon_loss
            add_global = global_step.assign_add(1)
            update_ops = tf.get_collection(tf.GraphKeys.UPDATE_OPS)
            with tf.control_dependencies([add_global] + update_ops):
                opt = solver.minimize(loss, var_list=Encoder.trainable_variables + Decoder.trainable_variables)
                with tf.control_dependencies([opt]):
                    return tf.identity(loss), tf.identity(recon_loss), \
                           tf.identity(kl_divergence)

        loss, r_loss, kl_loss = strategy.experimental_run_v2(train_step, (dataset.get_next()[0],))
        loss = strategy.reduce(tf.distribute.ReduceOp.MEAN, loss, axis=None)
        r_loss = strategy.reduce(tf.distribute.ReduceOp.MEAN, r_loss, axis=None)
        kl_loss = strategy.reduce(tf.distribute.ReduceOp.MEAN, kl_loss, axis=None)
        print("Building eval module...")
        fixed_z = tf.placeholder(tf.float32)
        random_z = tf.placeholder(tf.float32)
        fixed_interp_z0 = tf.placeholder(tf.float32)
        fixed_interp_z1 = tf.placeholder(tf.float32)
        random_interp_z0 = tf.placeholder(tf.float32)
        random_interp_z1 = tf.placeholder(tf.float32)

        fixed_img = tf.placeholder(tf.float32)
        random_img = tf.placeholder(tf.float32)
        fixed_interp_img0 = tf.placeholder(tf.float32)
        fixed_interp_img1 = tf.placeholder(tf.float32)
        random_interp_img0 = tf.placeholder(tf.float32)
        random_interp_img1 = tf.placeholder(tf.float32)

        fixed_gen_img, random_gen_img,\
        fixed_gen_interp_img, \
        random_gen_interp_img = generate_sample(Decoder, fixed_z, random_z, fixed_interp_z0,
                                                fixed_interp_z1, random_interp_z0, random_interp_z1,
                                                config.num_midpoints)
        fixed_recon_img, random_recon_img, \
        fixed_recon_interp_img, \
        random_recon_interp_img = reconstruction_sample(Encoder, Decoder, fixed_img, random_img, fixed_interp_img0,
                                                        fixed_interp_img1, random_interp_img0, random_interp_img1,
                                                        config.num_midpoints)

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
            fixed_z_ = np.random.normal(size=[config.batch_size, config.dim_z])
            fixed_interp_z0_ = np.random.normal(size=[config.batch_size, config.dim_z])
            fixed_interp_z1_ = np.random.normal(size=[config.batch_size, config.dim_z])
            print("Completing all work, iteration now start, consuming %s " % timer.runing_time_format)
            print("Start iterations...")
            for iteration in range(config.total_step):
                loss_, r_loss_, kl_loss_, lr_ = sess.run([loss, r_loss, kl_loss, learning_rate])
            if iteration % config.print_loss_per_steps == 0:
                timer.update()
                print("step %d, loss %f, r_loss_ %f, kl_loss_ %f, learning_rate % f, consuming time %s" %
                      (iteration, loss_, r_loss_, kl_loss_, lr_, timer.runing_time_format))
            if iteration % config.eval_per_steps == 0:
                pass
            if iteration % config.save_per_steps == 0:
                saver_e.save(sess, save_path=config.model_dir + '/en.ckpt',
                             global_step=iteration, write_meta_graph=False)
                saver_d.save(sess, save_path=config.model_dir + '/de.ckpt',
                             global_step=iteration, write_meta_graph=False)





