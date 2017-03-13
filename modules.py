'''
Thanks and credit to xternalz for non-librarian implementation:
https://github.com/xternalz/WideResNet-pytorch/blob/master/wideresnet.py
'''

import math
import torch
import torch.nn as nn
import torch.nn.functional as F
from torch.autograd import Variable


def shape_like(tensor1, tensor2):
    tensor1, tensor2 = pool_shape_like(tensor1, tensor2)
    tensor1, tensor2 = channel_shape_like(tensor1, tensor2)
    return tensor1, tensor2

def pool_shape_like(tensor1, tensor2):
    ratio = tensor1.size(2) / tensor2.size(2)
    if ratio > 1:
        while ratio > 1:
            tensor1 = F.avg_pool2d(tensor1, 2)
            if ratio % 2 is not 0:
                raise Exception("Not a power of 2 ratio between tensor sizes")
            ratio = tensor1.size(2) / tensor2.size(2)

    elif ratio < 1:
        ratio = tensor2.size(2) / tensor1.size(2)
        while ratio > 1:
            tensor2 = F.avg_pool2d(tensor2, 2)
            if ratio % 2 is not 0:
                raise Exception("Not a power of 2 ratio between tensor sizes")
            ratio = tensor2.size(2) / tensor1.size(2)

    return tensor1, tensor2

def channel_shape_like(tensor1, tensor2):
    if tensor1.size(1) > tensor2.size(1):
        module_channels = tensor1.size(1)
        shaper = ChannelShaper(module_channels)
        tensor2 = shaper.shape(tensor2)
    else:
        module_channels = tensor2.size(1)
        shaper = ChannelShaper(module_channels)
        tensor1 = shaper.shape(tensor1)

    return tensor1, tensor2

class ChannelShaper(object):
    def __init__(self, module_channels, output_channels=None):
        self.module_channels = module_channels
        self.input_channels = None
        self.to_restore = None
        self.output_channels = output_channels if output_channels is not None else module_channels

    def shape(self, inputs):
        self.input_channels = input_channels = inputs.size()[1]
        if input_channels < self.module_channels:
            sizes = [s for s in inputs.size()]
            sizes[1] = self.module_channels - sizes[1]
            zeros = Variable(torch.zeros(sizes[0], sizes[1], sizes[2], sizes[3]))
            # print(sizes)
            # print(zeros.size())
            output = torch.cat((inputs, zeros), 1)
            return output

        elif input_channels > self.module_channels:
            output = inputs[:, 0:self.module_channels]
            self.to_restore = inputs[:, self.module_channels:]
            return output

        else:
            return inputs

    def unshape(self, inputs, restore_dims=False):
        if self.input_channels < self.module_channels:
            return inputs[:, :self.input_channels]

        elif self.input_channels > self.module_channels and restore_dims:
            if self.to_restore is None:
                raise Exception("No saved tensor to restore lost channels")

            elif self.to_restore.size(1) is not (self.input_channels - self.module_channels):
                raise Exception("Saved tensor to restore lost channels was wrong size")

        else:
            return inputs

class Skipper(nn.Module):
    def __init__(self):
        super(Skipper, self).__init__()
        self.saved_input = None

    def forward(self, x):
        if self.saved_input is None:
            self.saved_input = x
            return x

        else:
            x1, x2 = shape_like(self.saved_input, x)
            return x1 + x2
###
class BN(nn.Module):
    def __init__(self, max_planes):
        super(BN, self).__init__()
        self.shaper = ChannelShaper(max_planes)
        self.bn = nn.BatchNorm2d(max_planes)

    def forward(self, x):
        x = self.shaper.shape(x)
        x = self.bn(x)
        x = self.shaper.unshape(x, restore_dims=True)
        return x

class Classifier(nn.Module):
    def __init__(self, num_classes):
        super(Classifier, self).__init__()
        self.num_classes = num_classes
        self.bottle = ForceChannels(num_classes)

    def forward(self, x):
        x = self.bottle(x)
        x = F.avg_pool2d(x, (x.size(2), x.size(3)))
        x = x.view(-1, self.num_classes)
        return F.log_softmax(x)

class ForceChannels(nn.Module):
    def __init__(self, num_channels):
        super(ForceChannels, self).__init__()
        self.shaper = ChannelShaper(num_channels)

    def forward(self, x):
        x = self.shaper.shape(x)
        return x

class Conv(nn.Module):
    def __init__(self, in_planes, out_planes):
        super(Conv, self).__init__()
        self.shaper = ChannelShaper(in_planes)
        self.conv = nn.Conv2d(in_planes, out_planes, 3)

    def forward(self, x):
        x = self.shaper.shape(x)
        x = self.shaper.unshape(self.conv(x))
        return x

class InitialConv(nn.Module):
    def __init__(self, in_channels):
        super(InitialConv, self).__init__()
        self.conv = nn.Conv2d(in_channels, 40, 3)

    def forward(self, x):
        return self.conv(x)

class Conv40(Conv):
    def __init__(self):
        super(Conv40, self).__init__(40, 40)

class Conv40_20(Conv):
    def __init__(self):
        super(Conv40_20, self).__init__(40, 20)

class Conv20(Conv):
    def __init__(self):
        super(Conv20, self).__init__(20, 20)

class Conv20_10(Conv):
    def __init__(self):
        super(Conv20_10, self).__init__(20, 10)

class ReLU(nn.ReLU):
    def __init__(self):
        super(ReLU, self).__init__(inplace=True)
