import torch
import torch.nn as nn
import torch.optim as optim
import torch.nn.functional as F

from math import floor, ceil


class ModuleSet(object):
    def __init__(self):
        self.set = {}

        self.add('convlstm', ConvLSTM)
        self.add('flatten', nn.Flatten)

        self.add('maxpool', nn.MaxPool2d)
        self.add('batchnorm1d', nn.BatchNorm1d)
        self.add('batchnorm2d', nn.BatchNorm2d)

        self.add('elu', nn.ELU)
        self.add('hardshrink', nn.Hardshrink)
        self.add('hardsigmoid', nn.Hardsigmoid)
        self.add('hardtanh', nn.Hardtanh)
        self.add('hardtanh', nn.Hardtanh)
        self.add('hardswish', nn.Hardswish)
        self.add('leakyrelu', nn.LeakyReLU)
        self.add('logsigmoid', nn.LogSigmoid)
        self.add('multiheadattention', nn.MultiheadAttention)
        self.add('prelu', nn.PReLU)
        self.add('relu', nn.ReLU)
        self.add('relu6', nn.ReLU6)
        self.add('rrelu', nn.RReLU)
        self.add('selu', nn.SELU)
        self.add('celu', nn.CELU)
        self.add('gelu', nn.GELU)
        self.add('sigmoid', nn.Sigmoid)
        self.add('softplus', nn.Softplus)
        self.add('softshrink', nn.Softshrink)
        self.add('softsign', nn.Softsign)
        self.add('tanh', nn.Tanh)
        self.add('tanhshrink', nn.Tanhshrink)
        self.add('threshold', nn.Threshold)

        self.add('softmin', nn.Softmin)
        self.add('softmax', nn.Softmax)
        self.add('softmax2d', nn.Softmax2d)
        self.add('logsoftmax', nn.LogSoftmax)
        self.add('adaptivelogsoftmaxwithloss', nn.AdaptiveLogSoftmaxWithLoss)

    def add(self, name, function):
        self.set[name] = function

    def get(self, name):
        module = self.set.get(name)
        if module is None:
            raise InvalidLayer("No such layer type: {0!r}".format(name))
        return module

    def is_valid(self, name):
        return name in self.set


class LossFunctionSet(object):
    def __init__(self):
        self.set = {}
        self.add('crossentropy', nn.CrossEntropyLoss)
        self.add('bce', nn.BCELoss)

    def add(self, name, function):
        self.set[name] = function

    def get(self, name):
        module = self.set.get(name)
        if module is None:
            raise InvalidLayer("No such layer type: {0!r}".format(name))
        return module

    def is_valid(self, name):
        return name in self.set


class OptimizerSet(object):
    def __init__(self):
        self.set = {}
        self.add('adam', optim.Adam)

    def add(self, name, function):
        self.set[name] = function

    def get(self, name):
        module = self.set.get(name)
        if module is None:
            raise InvalidLayer("No such optimizer: {0!r}".format(name))
        return module

    def is_valid(self, name):
        return name in self.set


class ConvLSTM(nn.Module):
    def __init__(self, in_shape, kernal_size, am):
        super(ConvLSTM, self).__init__()

        self.in_channels = in_shape[1]
        self.out_channels = in_shape[1]
        self.kernal_size = kernal_size
        self.data_length = in_shape[-1]

        # kernal must be odd
        self.padding = int((self.kernal_size - 1) / 2)

        self.weights = nn.Conv2d(self.in_channels + self.out_channels,
                                 4 * self.out_channels,
                                 self.kernal_size, 1, self.padding)
        self.batchnorm = None
        if 'groupnorm' in am:
            self.batchnorm = nn.GroupNorm(4 * self.out_channels // 16, 4 * self.out_channels)

    def forward(self, inputs, seq_len=10):
        xs, h_c_state = inputs
        if h_c_state is None:
            h_in = torch.zeros(xs.size(1), self.out_channels,
                               self.data_length, self.data_length).cuda()
            c_in = torch.zeros(xs.size(1), self.out_channels,
                               self.data_length, self.data_length).cuda()
        else:
            h_in, c_in = h_c_state

        h_seq = []
        h_out = None
        c_out = None
        for index in range(seq_len):
            if xs is None:
                x = torch.zeros(h_in.size(0), self.in_channels,
                                self.data_length, self.data_length).cuda()
            else:
                x = xs[index, ...]

            combined = torch.cat((x, h_in), 1)
            gates = self.weights(combined)
            if self.batchnorm is not None:
                gates = self.batchnorm(gates)
            in_gate, forget_gate, cell_gate, out_gate = torch.split(gates, self.out_channels, dim=1)

            in_gate = torch.sigmoid(in_gate)
            forget_gate = torch.sigmoid(forget_gate)
            cell_gate = torch.tanh(cell_gate)
            out_gate = torch.sigmoid(out_gate)

            c_out = (forget_gate * c_in) + (in_gate * cell_gate)
            h_out = out_gate * torch.tanh(c_out)
            h_seq.append(h_out)
            h_in = h_out
            c_in = c_out

        h_c_state = (h_out, c_out)
        h_seq = torch.stack(h_seq)

        return h_seq, h_c_state


class ConvinLSTM(nn.Module):
    def __init__(self, in_channels, out_channels, kernal_size, stride, padding):
        super(ConvinLSTM, self).__init__()

        self.weigths = nn.Conv2d(in_channels, out_channels, kernal_size, stride, padding, bias=True)

    def forward(self, inputs):
        seq_number, batch_size, data_channels, data_length, _ = inputs.size()
        inputs = torch.reshape(inputs, (-1, data_channels, data_length, data_length))
        out = self.weigths(inputs)
        out = torch.reshape(out, (seq_number, batch_size, out.size(1), out.size(2), out.size(3)))
        return out


class ConvTransinLSTM(nn.Module):
    def __init__(self, in_channels, out_channels, kernal_size, stride, padding, bias=True):
        super(ConvTransinLSTM, self).__init__()

        self.weigths = nn.ConvTranspose2d(in_channels, out_channels, kernal_size, stride, padding, bias=bias)

    def forward(self, inputs):
        seq_number, batch_size, data_channels, data_length, _ = inputs.size()
        inputs = torch.reshape(inputs, (-1, data_channels, data_length, data_length))
        out = self.weigths(inputs)
        out = torch.reshape(out, (seq_number, batch_size, out.size(1), out.size(2), out.size(3)))
        return out


class ResModule(nn.Module):
    def __init__(self, in_channels, out_channels,
                 kernel_size, stride, padding, is_empty_skip, init_mode=1):
        super(ResModule, self).__init__()

        self.init_mode = init_mode

        self.conv1 = nn.Sequential(
            nn.Conv2d(in_channels, out_channels,
                      kernel_size=kernel_size, stride=stride, padding=padding, bias=False),
            nn.BatchNorm2d(out_channels),
        )
        self.conv1.apply(self.init_weights)

        self.conv2 = nn.Sequential(
            nn.Conv2d(out_channels, out_channels,
                      kernel_size=kernel_size, stride=1, padding=int((kernel_size-1)/2), bias=False),
            nn.BatchNorm2d(out_channels),
        )
        self.conv2.apply(self.init_weights)

        if is_empty_skip:
            self.shortcut = nn.Sequential()
        else:
            self.shortcut = nn.Sequential(
                nn.Conv2d(in_channels, out_channels,
                          kernel_size=kernel_size, stride=stride, padding=padding, bias=False),
                nn.BatchNorm2d(out_channels),
            )
            self.shortcut.apply(self.init_weights)

    def forward(self, x):
        out = F.relu(self.conv1(x))
        out = self.conv2(out)
        out += self.shortcut(x)
        out = F.relu(out)
        return out

    def init_weights(self, m):
        if isinstance(m, nn.Conv2d):
            if self.init_mode == 1:
                nn.init.kaiming_normal_(m.weight.data)
