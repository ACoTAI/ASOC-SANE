import os
import yaml
import torch

from execution import SANEExecuion
import torchvision.datasets as datasets
import torchvision.transforms as transforms
from data.MovingMNIST import MovingMNIST


if __name__ == '__main__':
    local_dir = os.path.dirname(__file__)
    config_file = os.path.join(local_dir, 'evolve_resnet_cifar10_config.yaml')

    with open(config_file, encoding='ascii', errors='ignore') as f:
        evolve_config = yaml.safe_load(f)
    f.close()
    evolution_config = evolve_config['evolution_config']
    train_config = evolve_config['train_config']
    ann_config = evolve_config['ann_config']

    mnist_transform = transforms.Compose([
        transforms.ToTensor(),
        transforms.Normalize([0.5], [0.5]),
    ])

    cifar_transform = transforms.Compose([
        transforms.RandomCrop(32, padding=4),
        transforms.RandomHorizontalFlip(),
        transforms.ToTensor(),
        transforms.Normalize((0.4914, 0.4822, 0.4465), (0.2023, 0.1994, 0.2010)),
    ])

    celeba_transform = transforms.Compose([
        transforms.Resize(64),
        transforms.CenterCrop(64),
        transforms.ToTensor(),
        transforms.Normalize((0.5, 0.5, 0.5), (0.5, 0.5, 0.5)),
    ])

    data_train_loader = None
    data_val_loader = None

    if train_config['dataset'] == 'cifar10':
        cifar10_train = datasets.CIFAR10(root='./data', train=True, download=True, transform=cifar_transform)
        cifar10_val = datasets.CIFAR10(root='./data', train=False, download=True, transform=cifar_transform)
        data_train_loader = torch.utils.data.DataLoader(cifar10_train, batch_size=train_config['train_batches'],
                                                        shuffle=True, num_workers=0, pin_memory=True)
        data_val_loader = torch.utils.data.DataLoader(cifar10_val, batch_size=train_config['train_batches'],
                                                      shuffle=False, num_workers=0, pin_memory=True)

    elif train_config['dataset'] == 'cifar100':
        cifar100_train = datasets.CIFAR100(root='./data', train=True, download=True, transform=cifar_transform)
        cifar100_val = datasets.CIFAR100(root='./data', train=False, download=True, transform=cifar_transform)
        data_train_loader = torch.utils.data.DataLoader(cifar100_train, batch_size=train_config['train_batches'],
                                                        shuffle=True,
                                                        num_workers=4, pin_memory=True)
        data_val_loader = torch.utils.data.DataLoader(cifar100_val, batch_size=train_config['train_batches'], shuffle=False,
                                                      num_workers=4, pin_memory=True)

    elif train_config['dataset'] == 'celeba':
        celeba = datasets.ImageFolder(root='./data/celeba/', transform=celeba_transform)

        data_train_loader = torch.utils.data.DataLoader(celeba, batch_size=train_config.train_batches,
                                                        shuffle=True, num_workers=4)

    elif train_config['dataset'] == 'mnist':
        mnist_train = datasets.MNIST(root='./data', train=True, download=True, transform=mnist_transform)
        data_train_loader = torch.utils.data.DataLoader(mnist_train, batch_size=train_config['train_batches'],
                                                        shuffle=True, num_workers=4, pin_memory=True)

    elif train_config['dataset'] == 'fashion-mnist':
        fashion_train = datasets.FashionMNIST(root='./data', train=True, download=True, transform=mnist_transform)
        data_train_loader = torch.utils.data.DataLoader(fashion_train, batch_size=train_config['train_batches'],
                                                        shuffle=True, num_workers=4, pin_memory=True)

    elif train_config['dataset'] == 'moving-mnist':
        moving_mnist_train = MovingMNIST(root='data/', is_train=True,
                                         n_frames_input=10, n_frames_output=10)

        moving_mnist_valid = MovingMNIST(root='data/', is_train=False,
                                         n_frames_input=10, n_frames_output=10)

        data_train_loader = torch.utils.data.DataLoader(moving_mnist_train,
                                                        batch_size=train_config['train_batches'], shuffle=True)
        data_val_loader = torch.utils.data.DataLoader(moving_mnist_valid,
                                                      batch_size=train_config['train_batches'], shuffle=True)

    evolve_execution = SANEExecuion(evolution_config, train_config, ann_config)
    evolve_execution.evolve_population(data_train_loader, data_val_loader)

    print('done!')
