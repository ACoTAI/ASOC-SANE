import torch
import torch.optim as optim
from torchvision.utils import save_image
from torch_fidelity import calculate_metrics
from components import LossFunctionSet, OptimizerSet


class TrainANN(object):
    def __init__(self, train_config, ann_type, phenotype):
        self.device = torch.device(train_config['train_device'] if torch.cuda.is_available() else 'cpu')
        self.ann_type = ann_type
        self.train_config = train_config
        self.phenotype = phenotype.to(self.device)

        optimizers = OptimizerSet()
        loss_functions = LossFunctionSet()

        if self.ann_type == 'cnn':
            self.optimizer = optimizers.set[self.train_config['optimizer']](self.phenotype.parameters(),
                                                                            lr=self.train_config['learning_rate'])
            self.loss_fn = loss_functions.set[self.train_config['loss_function']]()
            self.scheduler = optim.lr_scheduler.CosineAnnealingLR(self.optimizer,
                                                                  T_max=self.train_config['full_train_epochs'])
        elif self.ann_type == 'gan':
            self.loss_fn = loss_functions.set[self.train_config['loss_function']]()
            self.latent_dim = self.train_config['input_size']['generator']

            self.optim_dict = {}
            self.scheduler_dict = {}
            for pheno_name in ['generator', 'discriminator']:
                self.phenotype.phenotype[pheno_name].to(self.device)
                self.optim_dict[pheno_name] = \
                    optimizers.set[self.train_config['optimizer']](self.phenotype.phenotype[pheno_name].parameters(),
                                                                   lr=self.train_config['learning_rate'],
                                                                   betas=(0.5, 0.999))
                self.scheduler_dict[pheno_name] = \
                    optim.lr_scheduler.CosineAnnealingLR(self.optim_dict[pheno_name],
                                                         T_max=self.train_config['full_train_epochs'])
        elif self.ann_type == 'lstm':
            self.optimizer = optimizers.set[self.train_config['optimizer']](self.phenotype.parameters(),
                                                                            lr=self.train_config['learning_rate'])
            self.loss_fn = loss_functions.set[self.train_config['loss_function']]()

    def train_ann(self, data_train, scheduler=None):
        if self.ann_type == 'cnn':
            self.train_cnn(data_train, scheduler)
        elif self.ann_type == 'gan':
            self.train_gan(data_train, scheduler)
        elif self.ann_type == 'lstm':
            self.train_lstm(data_train, scheduler)

    def valid_ann(self, data_valid=None):
        if self.ann_type == 'cnn':
            return self.valid_cnn(data_valid)
        elif self.ann_type == 'gan':
            return self.valid_gan(self.train_config['dataset'])
        elif self.ann_type == 'lstm':
            return self.valid_lstm(data_valid)

    def train_cnn(self, data_train, scheduler=False):
        if scheduler:
            epochs = self.train_config['full_train_epochs']
        else:
            epochs = self.train_config['train_epochs']

        for epoch in range(epochs):
            correct = 0
            total = 0
            for data, target in data_train:
                data = data.to(self.device)
                target = target.to(self.device)
                pred = self.phenotype(data)
                loss_batch = self.loss_fn(pred, target)

                self.optimizer.zero_grad()
                loss_batch.backward()
                self.optimizer.step()

                _, predict = torch.max(pred, 1)
                c = (predict == target)
                correct += sum(c)
                total += len(target)
            if scheduler:
                self.scheduler.step()
            accruracy = 100 * float(correct) / float(total)
            print(f'epoch {epoch} - accuracy {accruracy}')

    def valid_cnn(self, data_valid):
        self.phenotype.eval()
        with torch.no_grad():
            correct = 0
            total = 0
            for data, target in data_valid:
                data = data.to(self.device)
                target = target.to(self.device)
                outputs = self.phenotype(data)
                _, predicted = torch.max(outputs, 1)
                c = (predicted == target)
                correct += sum(c)
                total += len(target)
            accruracy = 100 * float(correct) / float(total)
            return accruracy

    def train_gan(self, data_train, scheduler=False):
        if scheduler:
            train_epochs = self.train_config['full_train_epochs']
        else:
            train_epochs = self.train_config['train_epochs']

        for epoch in range(train_epochs):
            for i, data in enumerate(data_train, 0):
                # train discriminator
                self.phenotype.phenotype['discriminator'].zero_grad()
                real_data = data[0].to(self.device)
                b_size = real_data.size(0)
                label = torch.full((b_size,), 1, dtype=torch.float, device=self.device)

                output = self.phenotype('discriminator', real_data).view(-1)
                errD_real = self.loss_fn(output, label)
                errD_real.backward()
                D_x = output.mean().item()

                noise = torch.randn(b_size, self.latent_dim[0], 1, 1, device=self.device)
                fake = self.phenotype('generator', noise)
                if len(fake.shape) == 2:
                    fake = fake.view(fake.size(0), 3, 64, 64)

                label.fill_(0)
                output = self.phenotype('discriminator', fake.detach()).view(-1)
                errD_fake = self.loss_fn(output, label)
                errD_fake.backward()
                D_G_z1 = output.mean().item()

                errD = errD_real + errD_fake
                self.optim_dict['discriminator'].step()

                # train generator
                self.phenotype.phenotype['generator'].zero_grad()
                label.fill_(1)

                output = self.phenotype('discriminator', fake).view(-1)
                errG = self.loss_fn(output, label)
                errG.backward()
                D_G_z2 = output.mean().item()
                self.optim_dict['generator'].step()

                if (i + 1) / len(data_train) == 1:
                    print('[%d/%d]\tLoss_D: %.4f\tLoss_G: %.4f\tD(x): %.4f\tD(G(z)): %.4f / %.4f' %
                          (epoch + 1, train_epochs, errD.item(), errG.item(), D_x, D_G_z1, D_G_z2))

            if scheduler:
                for organ in ['generator', 'discriminator']:
                    self.scheduler_dict[organ].step()

    def valid_gan(self, dataset):
        for i in range(10):
            noise = torch.randn(100, 100, 1, 1, device=self.device)
            self.phenotype.phenotype['generator'].eval()
            with torch.no_grad():
                generated_img = self.phenotype('generator', noise).detach().cpu()
                if len(generated_img.shape) == 2:
                    generated_img = generated_img.view(generated_img.size(0), 3, 64, 64)

            for j in range(i * 100 + 1, i * 100 + 101):
                if j < 10:
                    name = '0000' + str(j)
                elif j < 100:
                    name = '000' + str(j)
                elif j < 1000:
                    name = '00' + str(j)
                elif j < 10000:
                    name = '0' + str(j)
                else:
                    name = str(j)
                img_name = './data/generator/' + name + '.png'
                save_image(generated_img[j - i * 100 - 1], img_name, normalize=True)
        data_folder = './data/' + dataset + '-1000/'
        fid = calculate_metrics(input1='./data/generator/', input2=data_folder, cuda=True,
                                isc=False, fid=True, kid=False, verbose=False)
        fid_core = 500 - fid['frechet_inception_distance']
        return fid_core

    def train_lstm(self, data_train, scheduler=False):
        if scheduler:
            train_epochs = self.train_config['full_train_epochs']
        else:
            train_epochs = self.train_config['train_epochs']

        for epoch in range(train_epochs):
            train_loss = []
            for data, target in data_train:
                data = data.transpose(0, 1)
                data = data.to(self.device)
                target = target.to(self.device)

                pred = self.phenotype(data).transpose(0, 1)
                loss = self.loss_fn(pred, target)
                train_loss.append(loss)

                self.optimizer.zero_grad()
                loss.backward()
                self.optimizer.step()

            train_loss = torch.stack(train_loss)
            print(f'epoch {epoch} - training loss {torch.mean(train_loss)}')

        torch.cuda.empty_cache()

    def valid_lstm(self, data_valid):
        with torch.no_grad():
            valid_loss = []
            for data, target in data_valid:
                data = data.transpose(0, 1)
                data = data.to(self.device)
                target = target.to(self.device)

                pred = self.phenotype(data).transpose(0, 1)
                loss = self.loss_fn(pred, target)
                valid_loss.append(loss)

            fitness = torch.mean(torch.stack(valid_loss))
            print(f'valid loss {fitness}')
            torch.cuda.empty_cache()
            return float(1.0 - fitness)
