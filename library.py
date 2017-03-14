
import math

import numpy as np

import torch
import torch.nn as nn
import torch.nn.functional as F
from torch.autograd import Variable

import modules

class DynamicNetwork(nn.Module):
    def __init__(self, module_spec, input_channels, num_classes):
        super(DynamicNetwork, self).__init__()
        self.library = build_library(module_spec)
        self.initial_conv = modules.InitialConv(input_channels)
        self.classifier = modules.Classifier(num_classes)
        self.indices = []

    def set_structure(self, indices):
        self.indices = indices
        print("Setting structure.")
        print("Indices: " + ", ".join([str(i) for i in indices]))
        print("Module types: " + ", ".join([str(type(self.library[i])) for i in indices]))

    def forward(self, x):
        if self.indices is None:
            raise Exception("No network structure specified. Call set_structure")
        x = self.initial_conv(x)
        for i in self.indices:
            x = self.library[i](x)
        x = self.classifier(x)
        return x

def build_library(module_spec):
    l = []
    for module_name, number in module_spec:
        module_class = getattr(modules, module_name)
        l.append(module_class())
    library = nn.ModuleList(l)
    return library
