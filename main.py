from utils.utils import *
from argparse import ArgumentParser
from importlib import import_module

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
parser.add_argument(
    '--batch_size', type=int, default=256,
    help='Default overall batchsize (default: %(default)s)')
parser.add_argument(
    '--gpu_nums', type=int, default=1,
    help='Number of dataloader workers (default: %(default)s)')
parser.add_argument(
    '--img_shape', type=list, default=[1, 28, 28],
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
    '--laplace_lambda', type=float, default=1.0,
    help='Default overall batchsize (default: %(default)s)')
parser.add_argument(
    '--laplace_lambda_x', type=float, default=1.0,
    help='Default overall batchsize (default: %(default)s)')
parser.add_argument(
    '--sigma', type=float, default=1,
    help='Default overall batchsize (default: %(default)s)')
parser.add_argument(
    '--smooth_sigma', type=float, default=2.5,
    help='Default overall batchsize (default: %(default)s)')
parser.add_argument(
    '--smooth_sigma_x', type=float, default=2.5,
    help='Default overall batchsize (default: %(default)s)')

parser.add_argument(
    '--lr', type=float, default=0.0001,
    help='Default overall batchsize (default: %(default)s)')
parser.add_argument(
    '--decay_step', type=int, default=3000,
    help='Default overall batchsize (default: %(default)s)')
parser.add_argument(
    '--decay_coef', type=float, default=0.5,
    help='Default overall batchsize (default: %(default)s)')
parser.add_argument(
    '--beta2', type=float, default=0.999,
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
    '--restore_s_dir', type=str, default='/gdata/fengrl/ckpt/valina_vae/celeb_a/en.ckpt-104000',
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
    '--print_loss_per_steps', type=int, default=100,
    help='Use LZF compression? (default: %(default)s)')
parser.add_argument(
    '--model', type=str, default='vae',
    help='seed for np')
parser.add_argument(
    '--model_dir_root', type=str, default='/gdata/fengrl/vae',
    help='seed for np')
parser.add_argument(
    '--run_dir', type=str, default='/program/vae_lap',
    help='seed for np')


args = vars(parser.parse_args())
model = import_module('training_loop.' + args['model'])
tf.config.optimizer.set_jit(True)

config = Config()
config.set(**args)
config.make_task_dir()
config.make_task_log()
config.write_config()
model.training_loop(config=config)
config.terminate()
