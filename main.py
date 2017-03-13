
import math
import argparse
import numpy as np

import torch
import torch.nn as nn
import torch.nn.functional as F
from torch.autograd import Variable
import torch.optim as optim
from torchvision import datasets, transforms

from library import DynamicNetwork
import train_ac

'''
module_spec = [
("Conv40", 3),
("Conv40_20", 2),
("Conv20", 3),
("Conv20_10", 2),
("BN_40", 5),
("Skipper", 2),
("ReLU", 1)
]
'''
module_spec = [
("Conv40", 1),
("Conv40_20", 1),
("Conv20_10", 1),
("ReLU", 1)
]

parser = argparse.ArgumentParser(description='MNIST LibNet')
parser.add_argument('--batch-size', type=int, default=64, metavar='N',
                    help='input batch size for training (default: 64)')
parser.add_argument('--test-batch-size', type=int, default=1000, metavar='N',
                    help='input batch size for testing (default: 1000)')
parser.add_argument('--epochs', type=int, default=100, metavar='N',
                    help='number of epochs to train (default: 10)')
parser.add_argument('--batch_per_valid', type=int, default=1, metavar='N',
                    help='number of batches to train before resampling dynet (default: 1)')
parser.add_argument('--lr', type=float, default=0.01, metavar='LR',
                    help='learning rate (default: 0.01)')
parser.add_argument('--momentum', type=float, default=0.5, metavar='M',
                    help='SGD momentum (default: 0.5)')
parser.add_argument('--ac_lr', type=float, default=0.01, metavar='LR',
                    help='learning rate (default: 0.01)')
parser.add_argument('--ac_momentum', type=float, default=0.5, metavar='M',
                    help='SGD momentum (default: 0.5)')
parser.add_argument('--max-net-length', type=int, default=20, metavar='M',
                    help='maximum number of modules in dynet')
parser.add_argument('--no-cuda', action='store_true', default=False,
                    help='enables CUDA training')
parser.add_argument('--seed', type=int, default=1, metavar='S',
                    help='random seed (default: 1)')
parser.add_argument('--log-interval', type=int, default=10, metavar='N',
                    help='how many batches to wait before logging training status')
parser.add_argument('--embedding_size', type=int, default=128, metavar='ES',
                    help='Size of embedding for library')
parser.add_argument('--lstm_size', type=int, default=128, metavar='LS',
                    help='Size of lstm for actor-critic')
parser.add_argument('--num_steps', type=int, default=20, metavar='NS',
                    help='Max num steps for AC')
parser.add_argument('--dataset', type=str, default='MNIST', metavar='DS',
                    help='Dataset to use')
parser.add_argument('--valid_fraction', type=float, default=0.2, metavar='VF',
                    help='Fraction of train set to use for validation')
args = parser.parse_args()
args.cuda = not args.no_cuda and torch.cuda.is_available()

dynet = DynamicNetwork(module_spec, 1, 10)
if args.cuda:
    dynet.cuda()

train_ac.train(args, dynet)
