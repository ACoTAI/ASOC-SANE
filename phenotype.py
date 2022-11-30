import torch.nn as nn
from pheno import construct_conv_cell_pheno, construct_linear_cell_pheno, \
    construct_convtrans_cell_pheno, construct_convinlstm_cell_pheno, \
    construct_convlstm_cell_pheno, construct_convtransinlstm_cell_pheno, \
    construct_res_cell_pheno, construct_avgpool_pheno


def construct_ann(genotype, in_shape):
    if genotype.ann_config['ann_type'] == 'cnn':
        return CNNPhenotype(genotype, in_shape)
    elif genotype.ann_config['ann_type'] == 'gan':
        return GANPhenotype(genotype, in_shape)
    elif genotype.ann_config['ann_type'] == 'lstm':
        return LSTMPhenotype(genotype, in_shape)


class CNNPhenotype(nn.Module):
    def __init__(self, genotype, in_shape=None):
        super(CNNPhenotype, self).__init__()

        if in_shape is None:
            in_shape = [3, 32, 32]

        self.is_vaild = True
        self.phenotype = nn.Sequential()

        for cell_key in genotype.data_path['feature']:
            if self.is_vaild:
                cell = genotype.genotype_dict['feature'][cell_key]

                if cell['type'] == 'conv':
                    cell_pheno, in_shape = construct_conv_cell_pheno(cell, in_shape,
                                                                     genotype.ann_config['init_mode'])
                    if cell_pheno != -1:
                        self.phenotype.add_module('f-' + str(cell_key), cell_pheno)
                    else:
                        self.is_vaild = False

                elif cell['type'] == 'res':
                    cell_pheno, in_shape = construct_res_cell_pheno(cell, in_shape,
                                                                    genotype.ann_config['init_mode'])
                    if cell_pheno != -1:
                        self.phenotype.add_module('f-' + str(cell_key), cell_pheno)
                    else:
                        self.is_vaild = False

                elif cell['type'] == 'output':
                    if cell['cm'] == 'avg':
                        cell_pheno, in_shape = construct_avgpool_pheno(in_shape)
                        self.phenotype.add_module('f-'+str(cell_key)+'a', cell_pheno)
                    self.phenotype.add_module('f-' + str(cell_key), nn.Flatten())

        for cell_key in genotype.data_path['classifier']:
            if self.is_vaild:
                cell = genotype.genotype_dict['classifier'][cell_key]

                if cell['type'] in ['linear', 'output']:
                    cell_pheno, in_shape = construct_linear_cell_pheno(cell, in_shape)
                    if cell_pheno != -1:
                        self.phenotype.add_module('c-' + str(cell_key), cell_pheno)
                    else:
                        self.is_vaild = False

    def forward(self, x):
        return self.phenotype(x)


class GANPhenotype(nn.Module):
    def __init__(self, genotype, in_shape_dict):
        super(GANPhenotype, self).__init__()

        self.is_vaild = True

        self.phenotype = {'generator': nn.Sequential(), 'discriminator': nn.Sequential()}

        in_shape = in_shape_dict['generator']
        for cell_key in genotype.data_path['generator']:
            if self.is_vaild:
                cell = genotype.genotype_dict['generator'][cell_key]

                if cell['type'] == 'convtrans':
                    cell_pheno, in_shape = construct_convtrans_cell_pheno(cell, in_shape)
                    if cell_pheno != -1:
                        self.phenotype['generator'].add_module('g-' + str(cell_key), cell_pheno)
                    else:
                        self.is_vaild = False

                elif cell['type'] == 'output':
                    cell_pheno, in_shape = construct_convtrans_cell_pheno(cell, in_shape,
                                                                          in_shape_dict['discriminator'])
                    if cell_pheno != -1:
                        self.phenotype['generator'].add_module('g-' + str(cell_key), cell_pheno)
                    else:
                        self.is_vaild = False

        in_shape = in_shape_dict['discriminator']
        for cell_key in genotype.data_path['discriminator']:
            if self.is_vaild:
                cell = genotype.genotype_dict['discriminator'][cell_key]

                if cell['type'] == 'conv':
                    cell_pheno, in_shape = construct_conv_cell_pheno(cell, in_shape,
                                                                     genotype.ann_config['init_mode'])
                    if cell_pheno != -1:
                        self.phenotype['discriminator'].add_module('d-' + str(cell_key), cell_pheno)
                    else:
                        self.is_vaild = False

                if cell['type'] == 'output':
                    cell_pheno, in_shape = construct_conv_cell_pheno(cell, in_shape,
                                                                     genotype.ann_config['init_mode'],
                                                                     [1, 1, 1])

                    if cell_pheno != -1:
                        self.phenotype['discriminator'].add_module('d-' + str(cell_key), cell_pheno)
                    else:
                        self.is_vaild = False

    def forward(self, g_or_d, x):
        if g_or_d == 'generator':
            return self.phenotype['generator'](x)
        else:
            return self.phenotype['discriminator'](x)


class LSTMPhenotype(nn.Module):
    def __init__(self, genotype, in_shape):
        super(LSTMPhenotype, self).__init__()

        self.is_vaild = True

        self.h_c_dict = {}
        self.encoder_list = []
        self.decoder_list = []

        for cell_key in genotype.data_path['encoder']:
            if self.is_vaild:
                cell = genotype.genotype_dict['encoder'][cell_key]

                if cell['type'] == 'conv':
                    trans_out_shape = in_shape
                    cell_pheno, in_shape = construct_convinlstm_cell_pheno(cell, in_shape)
                    if cell_pheno != -1:
                        setattr(self, 'econv'+str(cell_key), cell_pheno)
                        self.encoder_list.append('econv'+str(cell_key))
                    else:
                        self.is_vaild = False

                    sigma = True if cell_key == 0 else False
                    cell_pheno, _ = construct_convtransinlstm_cell_pheno(cell, in_shape,
                                                                         trans_out_shape, sigma)
                    if cell_pheno != -1:
                        setattr(self, 'dconvtrans' + str(cell_key), cell_pheno)
                        self.decoder_list.append('dconvtrans' + str(cell_key))
                    else:
                        self.is_vaild = False

                elif cell['type'] == 'convlstm':
                    cell_pheno, in_shape = construct_convlstm_cell_pheno(cell, in_shape)
                    if cell_pheno != -1:
                        setattr(self, 'elstm'+str(cell_key), cell_pheno)
                        self.encoder_list.append('elstm'+str(cell_key))
                    else:
                        self.is_vaild = False

                    cell_pheno, _ = construct_convlstm_cell_pheno(cell, in_shape)
                    if cell_pheno != -1:
                        setattr(self, 'dlstm' + str(cell_key), cell_pheno)
                        self.decoder_list.append('dlstm' + str(cell_key))
                    else:
                        self.is_vaild = False

        self.decoder_list.reverse()
        self.phenotype = self.lstm_info()

    def forward(self, inputs):
        out = inputs
        for name in self.encoder_list:
            if name[1:5] == 'lstm':
                lstm_input = (out, None)
                out, h_c_state = getattr(self, name)(lstm_input)
                self.h_c_dict[name[1:]] = h_c_state
            else:
                out = getattr(self, name)(out)

        out = None
        for name in self.decoder_list:
            if name[1:5] == 'lstm':
                h_c_state_in_e = self.h_c_dict[name[1:]]
                lstm_input = (out, h_c_state_in_e)
                out, h_c_state = getattr(self, name)(lstm_input)
            else:
                out = getattr(self, name)(out)

        return out

    def lstm_info(self):
        lstm_info = []
        pheno_list = self.encoder_list + self.decoder_list
        for name in pheno_list:
            lstm_info.append((name, getattr(self, name)))
        return lstm_info


