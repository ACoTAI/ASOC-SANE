import torch.nn as nn
from math import floor, ceil

from components import ConvinLSTM, ConvTransinLSTM, ConvLSTM, ResModule


def construct_conv_cell_pheno(cell, in_shape, init_mode=1, out_shape=None):
    if out_shape is None:
        out_channels =  cell['cm'][1][0]
        kernel_size = cell['cm'][1][1]
        stride = cell['cm'][1][2]
        padding = cell['cm'][1][3]
        out_data_length = floor(float(in_shape[-1] + cell['cm'][1][3] * 2 - cell['cm'][1][1])
                                / float(cell['cm'][1][2]) + 1)
    else:
        out_channels = 1
        kernel_size = in_shape[-1]
        stride = 1
        padding = 0
        out_data_length = out_shape[-1]

    if out_data_length <= 0:
        return -1, out_data_length

    cell_pheno = nn.Sequential()

    if init_mode == 1:
        cm = nn.Conv2d(in_channels=in_shape[0], out_channels=out_channels,
                       kernel_size=kernel_size, stride=stride, padding=padding)
        nn.init.kaiming_normal_(cm.weight.data)
        if cm.bias is not None:
            cm.bias.data.zero_()

    else:
        cm = nn.Conv2d(in_channels=in_shape[0], out_channels=out_channels,
                       kernel_size=kernel_size, stride=stride, padding=padding,
                       bias=False)
        nn.init.normal_(cm.weight.data, 0.0, 0.02)

    cell_pheno.add_module('c', cm)

    if 'batchnorm' in cell.get('am'):
        am_batchnorm = nn.BatchNorm2d(cell['cm'][1][0])
        am_batchnorm.weight.data.fill_(1)
        am_batchnorm.bias.data.zero_()
        cell_pheno.add_module('b', am_batchnorm)

    if 'relu' in cell.get('am'):
        am_activation = nn.ReLU()
        cell_pheno.add_module('a', am_activation)
    elif 'leakyrelu' in cell.get('am'):
        am_activation = nn.LeakyReLU()
        cell_pheno.add_module('a', am_activation)
    elif 'sigmoid' in cell.get('am'):
        am_activation = nn.Sigmoid()
        cell_pheno.add_module('a', am_activation)

    if 'maxpool' in cell.get('am'):
        am_pool = nn.MaxPool2d(2)
        cell_pheno.add_module('p', am_pool)
        out_data_length = floor(float(out_data_length / 2))

        if out_data_length <= 0:
            return -1, out_data_length

    return cell_pheno, [out_channels, out_data_length, out_data_length]


def construct_linear_cell_pheno(cell, in_shape):
    cell_pheno = nn.Sequential()

    in_features = 1
    for i in in_shape:
        in_features *= i

    if in_features <= 0:
        return -1, in_shape

    cm = nn.Linear(in_features=in_features, out_features=cell['cm'][1])
    cm.weight.data.normal_(0, 0.01)
    cm.bias.data.zero_()
    cell_pheno.add_module('l', cm)

    if cell.get('am'):
        if 'relu' in cell.get('am'):
            cell_pheno.add_module('a', nn.ReLU())

    return cell_pheno, [cell['cm'][1]]


def construct_convtrans_cell_pheno(cell, in_shape, out_shape=None):
    if out_shape is None:
        kernel_size = cell['cm'][1][1]
        stride = cell['cm'][1][2]
        padding = cell['cm'][1][3]

        out_data_length = (in_shape[-1] - 1) * cell['cm'][1][2] - 2 * cell['cm'][1][3] + cell['cm'][1][1]

        out_shape = [cell['cm'][1][0], out_data_length, out_data_length]

    else:
        if in_shape[-1] > out_shape[-1]:
            if (in_shape[-1] - 1 - out_shape[-1]) % 2:
                kernel_size = 1
            else:
                kernel_size = 2
            padding = ceil((in_shape[-1] - 1 - out_shape[-1] + kernel_size) / 2)
        elif in_shape[-1] < out_shape[-1]:
            kernel_size = out_shape[-1] - in_shape[-1] + 1
            padding = 0
        else:
            kernel_size = 1
            padding = 0
        stride = 1

    if out_shape[-1] <= 0 or kernel_size <= 0 or stride <= 0 or padding < 0:
        return -1, out_shape

    cell_pheno = nn.Sequential()

    cm = nn.ConvTranspose2d(in_channels=in_shape[0], out_channels=out_shape[0],
                            kernel_size=kernel_size, stride=stride, padding=padding,
                            bias=False)
    nn.init.normal_(cm.weight.data, 0.0, 0.02)
    cell_pheno.add_module('c', cm)

    if 'batchnorm' in cell.get('am'):
        am_batchnorm = nn.BatchNorm2d(cell['cm'][1][0])
        nn.init.normal_(am_batchnorm.weight.data, 1.0, 0.02)
        nn.init.constant_(am_batchnorm.bias.data, 0)
        cell_pheno.add_module('b', am_batchnorm)

    if 'relu' in cell.get('am'):
        am_activation = nn.ReLU()
        cell_pheno.add_module('a', am_activation)
    elif 'tanh' in cell.get('am'):
        am_activation = nn.Tanh()
        cell_pheno.add_module('a', am_activation)

    return cell_pheno, out_shape


def construct_convinlstm_cell_pheno(cell, in_shape):
    out_channels = cell['cm'][1][0]
    kernel_size = cell['cm'][1][1]
    stride = cell['cm'][1][2]
    padding = cell['cm'][1][3]
    out_data_length = floor(float(in_shape[-1] + cell['cm'][1][3] * 2 - cell['cm'][1][1])
                            / float(cell['cm'][1][2]) + 1)
    out_shape = [in_shape[0], cell['cm'][1][0], out_data_length, out_data_length]

    if out_data_length <= 0:
        return -1, out_data_length

    cell_pheno = nn.Sequential()

    cm = ConvinLSTM(in_channels=in_shape[1], out_channels=out_channels,
                    kernal_size=kernel_size, stride=stride, padding=padding)
    cell_pheno.add_module('c', cm)

    if 'leakyrelu' in cell.get('am'):
        am_activation = nn.LeakyReLU()
        cell_pheno.add_module('a', am_activation)

    return cell_pheno, out_shape


def construct_convtransinlstm_cell_pheno(cell, in_shape, out_shape, sigma=False):
    out_channels = out_shape[1]
    stride = cell['cm'][1][2]
    padding = cell['cm'][1][3]
    kernel_size = int(out_shape[-1] + 2 * padding - (in_shape[-1]-1) * stride)

    if kernel_size <= 0 or stride <= 0 or padding < 0:
        return -1, out_shape

    cell_pheno = nn.Sequential()

    cm = ConvTransinLSTM(in_channels=in_shape[1], out_channels=out_channels,
                         kernal_size=kernel_size, stride=stride, padding=padding, bias=True)
    cell_pheno.add_module('tl', cm)

    if 'leakyrelu' in cell.get('am') and sigma is False:
        am_activation = nn.LeakyReLU()
        cell_pheno.add_module('a', am_activation)
    elif sigma:
        am_activation = nn.Sigmoid()
        cell_pheno.add_module('a', am_activation)

    return cell_pheno, out_shape


def construct_convlstm_cell_pheno(cell, in_shape, sigma=False):
    if not isinstance(in_shape, list):
        return -1, in_shape

    kernel_size = cell['cm'][1]
    am = cell.get('am')
    data_length = in_shape[-1]

    if data_length <= 0 or kernel_size > data_length:
        return -1, data_length

    cell_pheno = nn.Sequential()

    cm = ConvLSTM(in_shape=in_shape, kernal_size=kernel_size, am=am)
    cell_pheno.add_module('cl', cm)

    if sigma:
        am_activation = nn.Sigmoid()
        cell_pheno.add_module('a', am_activation)

    return cell_pheno, in_shape


def construct_res_cell_pheno(cell, in_shape, init_mode):
    out_channels = cell['cm'][1][0]
    kernel_size = cell['cm'][1][1]
    stride = cell['cm'][1][2]
    padding = cell['cm'][1][3]
    out_data_length = floor(float(in_shape[-1] + cell['cm'][1][3] * 2 - cell['cm'][1][1])
                            / float(cell['cm'][1][2]) + 1)

    if out_data_length <= 0:
        return -1, out_data_length

    cell_pheno = nn.Sequential()
    if in_shape == [out_channels, out_data_length, out_data_length]:
        is_empty_skip = True
    else:
        is_empty_skip = False

    cm = ResModule(in_channels=in_shape[0], out_channels=out_channels,
                   kernel_size=kernel_size, stride=stride, padding=padding,
                   is_empty_skip=is_empty_skip, init_mode=init_mode)

    cell_pheno.add_module('c', cm)

    return cell_pheno, [cell['cm'][1][0], out_data_length, out_data_length]


def construct_avgpool_pheno(in_shape):
    kernel_size = in_shape[-1]
    pheno = nn.AvgPool2d(kernel_size)
    return pheno, [in_shape[0], 1, 1]
