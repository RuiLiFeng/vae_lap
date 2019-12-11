from utils.utils import *
from network import wvae
import numpy as np


def training_loop(config:Config):
    timer = Timer()
    print('Task name : %s' % config.task_name)
    opts = {'dataset': config.dataset, "datashap": wvae.datashapes[config.dataset]}
    Encoder = wvae.Encoder(opts)
    Decoder = wvae.Decoder(opts)
