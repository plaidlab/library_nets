
import math

import numpy as np

import torch
import torch.nn as nn
import torch.nn.functional as F
from torch.autograd import Variable


class SimpleLibrarian(torch.nn.Module):
    def __init__(self, num_modules, embedding_size):
        super(SimpleLibrarian, self).__init__()
        self.embedding_size = embedding_size
        self.num_modules = num_modules
        self.embedding = Variable(torch.randn(self.num_modules, self.embedding_size))

    def forward(self, inputs):
        return self.embedding[inputs]
