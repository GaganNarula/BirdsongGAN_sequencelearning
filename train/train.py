from __future__ import print_function
import os
import argparse
import random
import torch
import torch.nn as nn
import torch.nn.parallel
import torch.backends.cudnn as cudnn
import torch.optim as optim
import torch.utils.data
import torchvision.datasets as dset
import torchvision.transforms as transforms
import torchvision.utils as vutils
import numpy as np
from torch.autograd import Variable
from datetime import datetime
import itertools
import shutil
from scipy.stats import t as t_dist
from models.networks_audio_nophase_8col import _netG,_netE,_netD,weights_init,GANLoss
import pickle
from utils.utils import *
import pdb


def make_output_folder(path):
    if not os.path.exists(path):
        os.makedirs(path)
    dirs = os.listdir(path)
    for d in dirs:
        if len(os.listdir(os.path.join(path, d))) <= 3:
            try:
                os.rmdir(os.path.join(path, d))
            except:
                shutil.rmtree(os.path.join(path, d))
    path += str(datetime.now()).replace(':', '-')
    if not os.path.exists(path):
        os.makedirs(path)
        os.makedirs(os.path.join(path, 'losses'))
        os.makedirs(os.path.join(path, 'hist'))
    return path


transform_threshold = 5.0
image_max_constant = 6.9236899002715671


opts = {'dataset' : 'folder',
        'dataroot' : 'input_path',
        'distance_fun' : 'L1',
        'workers' : 2,
        'batchSize' : 64,
        'imageH' : 129,
        'imageW' : 8,
        'nz' : 16,
        'nc' : 1,
        'ngf' : 256,
        'ndf' : 256,
        'niter' : niter,
        'lr' : 0.00001,
        'lambdaa' : 150,
        'd_noise' : 0.1,
        'beta1' : 0.5,
        'cuda' : 1,
        'ngpu' : 1,
        'netG' : '',
        'netD1' : '',
        'netD2' : '',
        'netD3' : '',
        'log_every' : 400,
        'sample_rate' : 16000.0,
        'outf' : '',
        'z_var' : noisevar,
        'nfft' : 256}


def 