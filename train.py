from __future__ import print_function
import torch
import torch.nn.functional as F
import torch.optim as optim
from torchvision import datasets, transforms
from torch.autograd import Variable
import torch.nn.init as init
import torch.nn as nn
from model import Net
from logger import Logger
import os


class trainer:
    def __init__(self, config):

        self.config = config
        self.load_data()
        self.load_model()
        self.logger = Logger('./logs/' + config.model_name)

    def load_data(self):
        kwargs = {'num_workers': 1, 'pin_memory': True} if self.config.cuda else {}
        self.train_loader = torch.utils.data.DataLoader(
            datasets.MNIST('../data', train=True, download=True,
                           transform=transforms.Compose([
                               transforms.ToTensor(),
                               transforms.Normalize((0.1307,), (0.3081,))
                           ])),
            batch_size=self.config.batch_size, shuffle=True, **kwargs)
        self.test_loader = torch.utils.data.DataLoader(
            datasets.MNIST('../data', train=False, transform=transforms.Compose([
                transforms.ToTensor(),
                transforms.Normalize((0.1307,), (0.3081,))
            ])),
            batch_size=self.config.batch_size, shuffle=True, **kwargs)

    def load_model(self):
        self.model = Net()

        if self.config.load_weights:
            self.load_weights(self.config.pretrained_weights)
        else:
            self.model.apply(self.weights_init)

        if self.config.cuda:
            self.model.cuda()

        self.optimizer = optim.SGD(self.model.parameters(), lr=self.config.lr, momentum=self.config.momentum)

    def train(self, epoch):
        self.model.train()
        for batch_idx, (data, target) in enumerate(self.train_loader):
            if self.config.cuda:
                data, target = data.cuda(), target.cuda()
            data, target = Variable(data), Variable(target)
            self.optimizer.zero_grad()
            output = self.model(data)
            loss = F.nll_loss(output, target)
            loss.backward()
            self.optimizer.step()
            if batch_idx % self.config.log_interval == 0:
                print('Train Epoch: {} [{}/{} ({:.0f}%)]\tLoss: {:.6f}'.format(
                    epoch, batch_idx * len(data), len(self.train_loader.dataset),
                    100. * batch_idx / len(self.train_loader), loss.data[0]))

        self.save_weights(self.config.model_name, epoch)
        train_loss, train_acc = self.test(self.train_loader)
        test_loss, test_acc = self.test(self.test_loader)
        self.tensorboard_logging(train_loss, test_loss, train_acc, test_acc, epoch)

    def test(self, loader):
        self.model.eval()
        test_loss = 0
        correct = 0
        for data, target in loader:
            if self.config.cuda:
                data, target = data.cuda(), target.cuda()
            data, target = Variable(data, volatile=True), Variable(target)
            output = self.model(data)
            test_loss += F.nll_loss(output, target, size_average=False).data[0] # sum up batch loss
            pred = output.data.max(1, keepdim=True)[1] # get the index of the max log-probability
            correct += pred.eq(target.data.view_as(pred)).cpu().sum()

        test_loss /= len(loader.dataset)
        print('\nEvaluation: Average loss: {:.4f}, Accuracy: {}/{} ({:.0f}%)\n'.format(
            test_loss, correct, len(loader.dataset),
            100. * correct / len(loader.dataset)))

        self.model.train()
        return test_loss, 100. * correct / len(loader.dataset)

    def weights_init(self, module):
        if isinstance(module, nn.Linear):
            init.xavier_normal(module.weight.data)
            init.constant(module.bias.data, 0.0)

    def save_weights(self, model_name, epoch):
        directory = './checkpoints/{}'.format(model_name)
        if not os.path.exists(directory):
            os.makedirs(directory)
        path = directory + '/epoch_{}_model.pth'.format(epoch)
        torch.save(self.model.state_dict(), path)

    def load_weights(self, path):
        self.model.load_state_dict(torch.load(path))

    def tensorboard_logging(self, train_loss, test_loss, train_acc, test_acc, epoch):
        info = {
            'train loss': train_loss,
            'test loss': test_loss,
            'train accuracy': train_acc,
            'test accuracy': test_acc
        }

        for tag, value in info.items():
            self.logger.scalar_summary(tag, value, epoch)

        # (2) Log values and gradients of the parameters (histogram)
        for tag, value in self.model.named_parameters():
            if 'cnn.fc' in tag or 'cnn' not in tag:
                tag = tag.replace('.', '/')
                self.logger.histo_summary(tag, to_np(value), epoch)
                if value.grad is not None:
                    self.logger.histo_summary(tag + '/grad', to_np(value.grad), epoch)

def to_np(x):
    return x.data.cpu().numpy()