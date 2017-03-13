'''Actor Critic model/algorithm for library nets.

Based on OpenAI Universe starter agent:
https://github.com/openai/universe-starter-agent/blob/master/a3c.py

And PyTorch port:
https://github.com/ikostrikov/pytorch-a3c/blob/master/model.py
'''

import math

import numpy as np

import torch
import torch.nn as nn
import torch.nn.functional as F
from torch.autograd import Variable


def normalized_columns_initializer(weights, std=1.0):
    out = torch.randn(weights.size())
    out *= std / torch.sqrt(out.pow(2).sum(1).expand_as(out))
    return out


def weights_init(m):
    classname = m.__class__.__name__
    if classname.find('Conv') != -1:
        weight_shape = list(m.weight.data.size())
        fan_in = np.prod(weight_shape[1:4])
        fan_out = np.prod(weight_shape[2:4]) * weight_shape[0]
        w_bound = np.sqrt(6. / (fan_in + fan_out))
        m.weight.data.uniform_(-w_bound, w_bound)
        m.bias.data.fill_(0)
    elif classname.find('Linear') != -1:
        weight_shape = list(m.weight.data.size())
        fan_in = weight_shape[1]
        fan_out = weight_shape[0]
        w_bound = np.sqrt(6. / (fan_in + fan_out))
        m.weight.data.uniform_(-w_bound, w_bound)
        m.bias.data.fill_(0)



class ActorCritic(torch.nn.Module):

    def __init__(self, action_space, librarian, lstm_size):
        super(ActorCritic, self).__init__()

        self.librarian = librarian

        print("Librarian type, " + str(type(librarian)))

        self.lstm = nn.LSTMCell(librarian.embedding_size, lstm_size)

        num_outputs = action_space
        self.critic_linear = nn.Linear(lstm_size, 1)
        self.actor_linear = nn.Linear(lstm_size, num_outputs)

        self.apply(weights_init)
        self.actor_linear.weight.data = normalized_columns_initializer(
            self.actor_linear.weight.data, 0.01)
        self.actor_linear.bias.data.fill_(0)
        self.critic_linear.weight.data = normalized_columns_initializer(
            self.critic_linear.weight.data, 1.0)
        self.critic_linear.bias.data.fill_(0)

        self.lstm.bias_ih.data.fill_(0)
        self.lstm.bias_hh.data.fill_(0)

        self.train()

    def forward(self, inputs, hx_cx):
        hx, cx = hx_cx

        x = inputs.view(-1, self.librarian.embedding_size)
        hx, cx = self.lstm(x, (hx, cx))
        x = hx

        return self.critic_linear(x), self.actor_linear(x), (hx, cx)
