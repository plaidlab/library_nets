import math

import torch.optim as optim

import numpy as np

import torch
import torch.nn as nn
import torch.nn.functional as F
from torch.autograd import Variable
from torchvision import datasets, transforms

import modules

class Environment(object):

    def __init__(self, args, dynet, libr):
        self.dynet = dynet
        self.libr = libr
        self.num_actions = len(dynet.library) + 1
        self.embedding_size = libr.embedding_size
        self.zero_state = Variable(torch.zeros(1, args.lstm_size))

        self.epoch = 0

        self.args = args

        self.prep_data()

        self.last_valid_accuracy = 0

        self.module_seq = []

        self.optimizer = optim.SGD(self.dynet.parameters(), lr=args.lr, momentum=args.momentum)



    def reset(self):
        self.module_seq = []
        return self.zero_state

    def step(self, action):

        if (action < self.num_actions - 1) and (action >= 0):
            self.module_seq.append(action)
            state = self.libr(action[0, 0])
            reward = 0
            done = False
            return state, reward, done, None

        elif action == self.num_actions - 1:
            reward = self.train_fixed_model()
            state = self.libr(action)
            done = True
            return state, reward, done, None

        else:
            raise Exception("Action out of range")


    def train_fixed_model(self):

        self.dynet.set_structure(self.module_seq)
        self.dynet.train()
        n_batches = len(self.train_loader)
        train_batches = n_batches * (1 - self.args.valid_fraction)
        valid_accuracies = []
        test_loss = 0
        correct = 0

        for batch_idx, (data, target) in enumerate(self.train_loader):

            if batch_idx <= train_batches:

                self.dynet.train()
                data, target = Variable(data), Variable(target)
                if self.args.cuda:
                    data, target = data.cuda(), target.cuda()

                self.optimizer.zero_grad()
                output = self.dynet(data)
                loss = F.nll_loss(output, target)
                loss.backward()
                self.optimizer.step()

                if batch_idx % self.args.log_interval == 0:
                    print('Train Epoch: {} [{}/{} ({:.0f}%)]\tLoss: {:.6f}'.format(
                        self.epoch, batch_idx * len(data) * train_batches / n_batches,
                        len(self.train_loader.dataset) * train_batches / n_batches,
                        100. * batch_idx / train_batches, loss.data[0]))
            else:
                self.dynet.eval()
                data, target = Variable(data, volatile=True), Variable(target)
                if self.args.cuda:
                    data, target = data.cuda(), target.cuda()

                output = self.dynet(data)
                test_loss += F.nll_loss(output, target).data[0]
                pred = output.data.max(1)[1] # get the index of the max log-probability
                correct += pred.eq(target.data).cpu().sum()

        accuracy = correct / (len(self.train_loader.dataset) * (n_batches - train_batches) / n_batches)

        print('Validation after Train Epoch: {} [{}/{} ({:.0f}%)]\tLoss: {:.6f}\tAccuracy: {:.6f}'.format(
            self.epoch, batch_idx * len(data) * (n_batches - train_batches) / n_batches,
            len(self.train_loader.dataset) * (n_batches - train_batches) / n_batches,
            100. * batch_idx / (n_batches - train_batches), loss.data[0]),
            correct / (len(self.train_loader.dataset) * (n_batches - train_batches) / n_batches))

        if self.epoch % self.args.test_epochs == 0:
            self.test_model()

        return accuracy

    def test_model(self):
        model.eval()
        test_loss = 0
        correct = 0
        for data, target in self.test_loader:
            if self.args.cuda:
                data, target = data.cuda(), target.cuda()
            data, target = Variable(data, volatile=True), Variable(target)
            output = self.dynet(data)
            test_loss += F.nll_loss(output, target).data[0]
            pred = output.data.max(1)[1] # get the index of the max log-probability
            correct += pred.eq(target.data).cpu().sum()

        test_loss = test_loss
        test_loss /= len(test_loader) # loss function already averages over batch size
        print('\nTest set: Average loss: {:.4f}, Accuracy: {}/{} ({:.0f}%)\n'.format(
            test_loss, correct, len(test_loader.dataset),
            100. * correct / len(test_loader.dataset)))

    def prep_data(self):
        # The output of torchvision datasets are PILImage images of range [0, 1].
        # We transform them to Tensors of normalized range [-1, 1]
        if self.args.dataset == 'MNIST':

            kwargs = {'num_workers': 1, 'pin_memory': True} if self.args.cuda else {}

            self.train_loader = torch.utils.data.DataLoader(
                datasets.MNIST('../data', train=True, download=True,
                               transform=transforms.Compose([
                                   transforms.ToTensor(),
                                   transforms.Normalize((0.1307,), (0.3081,))
                               ])),
                batch_size=self.args.batch_size, shuffle=True, **kwargs)

            self.test_loader = torch.utils.data.DataLoader(
                datasets.MNIST('../data', train=False, transform=transforms.Compose([
                                   transforms.ToTensor(),
                                   transforms.Normalize((0.1307,), (0.3081,))
                               ])),
                batch_size=self.args.batch_size, shuffle=True, **kwargs)

        else:
            raise Exception("Must specify a supported dataset")
