# This implementation is based on the DenseNet-BC implementation in torchvision
# https://github.com/pytorch/vision/blob/master/torchvision/models/densenet.py

import torch
import torch.nn as nn
import torch.nn.functional as F
from collections import OrderedDict

is_mobile = True
is_inception = False

class _DenseLayer(nn.Sequential):
    def __init__(self, num_input_features, growth_rate, bn_size, drop_rate):
        super(_DenseLayer, self).__init__()
        
#         mid = int((num_input_features + growth_rate) / 2)
        mid = growth_rate
        
        self.add_module('norm.1', nn.BatchNorm2d(num_input_features)),
        self.add_module('relu.1', nn.ReLU(inplace=True)),
        if not is_mobile and not is_inception:
            self.add_module('conv.1', nn.Conv2d(num_input_features,
                          growth_rate, kernel_size=3, stride=1, padding=1, bias=False)),
        
        elif is_inception:
            # inception
            self.add_module('conv.1', nn.Conv2d(num_input_features,
                             mid, kernel_size=(3, 1), stride=1, padding=(1, 0), bias=False)),

            self.add_module('norm.2', nn.BatchNorm2d(mid)),
            self.add_module('relu.2', nn.ReLU(inplace=True)),
            self.add_module('conv.2', nn.Conv2d(mid,
                             growth_rate, kernel_size=(1, 3), stride=1, padding=(0, 1), bias=False)),

        elif is_mobile:
            # mobile
            self.add_module('conv.1', nn.Conv2d(num_input_features,
                             num_input_features, groups=num_input_features, kernel_size=3, stride=1, padding=1, bias=False)),

            self.add_module('norm.2', nn.BatchNorm2d(num_input_features)),
            self.add_module('relu.2', nn.ReLU(inplace=True)),
            self.add_module('conv.2', nn.Conv2d(num_input_features,
                             growth_rate, kernel_size=1, stride=1, padding=0, bias=False)),
        
        
        
#         self.add_module('conv.1', nn.Conv2d(num_input_features, bn_size *
#                         growth_rate, kernel_size=1, stride=1, bias=False)),
#         self.add_module('norm.2', nn.BatchNorm2d(bn_size * growth_rate)),
#         self.add_module('relu.2', nn.ReLU(inplace=True)),
#         self.add_module('conv.2', nn.Conv2d(bn_size * growth_rate, growth_rate,
#                         kernel_size=3, stride=1, padding=1, bias=False)),
        self.drop_rate = drop_rate

    def forward(self, x):
        new_features = super(_DenseLayer, self).forward(x)
        if self.drop_rate > 0:
            new_features = F.dropout(new_features, p=self.drop_rate, training=self.training)
        return torch.cat([x, new_features], 1)


class _Transition(nn.Sequential):
    def __init__(self, num_input_features, num_output_features):
        super(_Transition, self).__init__()
        self.add_module('norm', nn.BatchNorm2d(num_input_features))
        self.add_module('relu', nn.ReLU(inplace=True))
        self.add_module('conv', nn.Conv2d(num_input_features, num_output_features,
                                          kernel_size=1, stride=1, bias=False))
        self.add_module('pool', nn.AvgPool2d(kernel_size=2, stride=2))
        # self.add_module('pool', nn.MaxPool2d(kernel_size=2, stride=2))


class _DenseBlock(nn.Sequential):
    def __init__(self, num_layers, num_input_features, bn_size, growth_rate, drop_rate):
        super(_DenseBlock, self).__init__()
        for i in range(num_layers):
            layer = _DenseLayer(num_input_features + i * growth_rate, growth_rate, bn_size, drop_rate)
            self.add_module('denselayer%d' % (i + 1), layer)


class DenseNet(nn.Module):
    r"""Densenet-BC model class, based on
    `"Densely Connected Convolutional Networks" <https://arxiv.org/pdf/1608.06993.pdf>`
    Args:
        growth_rate (int) - how many filters to add each layer (`k` in paper)
        block_config (list of 3 or 4 ints) - how many layers in each pooling block
        num_init_features (int) - the number of filters to learn in the first convolution layer
        bn_size (int) - multiplicative factor for number of bottle neck layers
            (i.e. bn_size * k features in the bottleneck layer)
        drop_rate (float) - dropout rate after each dense layer
        num_classes (int) - number of classification classes
        small_inputs (bool) - set to True if images are 32x32. Otherwise assumes images are larger.
    """
    # def __init__(self, growth_rate=12, block_config=(16, 16, 16), compression=0.5,
    def __init__(self, growth_rate=12, block_config=(16, 16, 16), compression=1,
                 num_init_features=24, bn_size=4, drop_rate=0,
                 num_classes=10, small_inputs=True):

        super(DenseNet, self).__init__()
        assert 0 < compression <= 1, 'compression of densenet should be between 0 and 1'
        self.avgpool_size = 8 if small_inputs else 7
        print("compression:", compression)
        # First convolution
        if small_inputs:
            self.features = nn.Sequential(OrderedDict([
                ('conv0', nn.Conv2d(3, num_init_features, kernel_size=3, stride=1, padding=1, bias=False)),
            ]))
        else:
            self.features = nn.Sequential(OrderedDict([
                ('conv0', nn.Conv2d(3, num_init_features, kernel_size=7, stride=2, padding=3, bias=False)),
            ]))
            self.features.add_module('norm0', nn.BatchNorm2d(num_init_features))
            self.features.add_module('relu0', nn.ReLU(inplace=True))
            self.features.add_module('pool0', nn.MaxPool2d(kernel_size=3, stride=2, padding=1,
                                                           ceil_mode=False))

        # Each denseblock
        num_features = num_init_features
        for i, num_layers in enumerate(block_config):
            block = _DenseBlock(num_layers=num_layers,
                                num_input_features=num_features,
                                bn_size=bn_size, growth_rate=growth_rate,
                                drop_rate=drop_rate)
            self.features.add_module('denseblock%d' % (i + 1), block)
            num_features = num_features + num_layers * growth_rate
            if i != len(block_config) - 1:
                trans = _Transition(num_input_features=num_features,
                                    num_output_features=int(num_features * compression))
                self.features.add_module('transition%d' % (i + 1), trans)
                num_features = int(num_features * compression)

        # Final batch norm
        self.features.add_module('norm_final', nn.BatchNorm2d(num_features))

        # Linear layer
        self.classifier = nn.Linear(num_features, num_classes)

    def forward(self, x):
        features = self.features(x)
        out = F.relu(features, inplace=True)
        out = F.avg_pool2d(out, kernel_size=self.avgpool_size).view(features.size(0), -1)
        out = self.classifier(out)
        return out

# import fire
import os
import time
import torch
from torchvision import datasets, transforms
# from models import DenseNet, DenseNetEfficient

result_train = []
result_test = []

class AverageMeter(object):
    """
    Computes and stores the average and current value
    Copied from: https://github.com/pytorch/examples/blob/master/imagenet/main.py
    """
    def __init__(self):
        self.reset()

    def reset(self):
        self.val = 0
        self.avg = 0
        self.sum = 0
        self.count = 0

    def update(self, val, n=1):
        self.val = val
        self.sum += val * n
        self.count += n
        self.avg = self.sum / self.count


def train_epoch(model, loader, optimizer, epoch, n_epochs, print_freq=1):
    batch_time = AverageMeter()
    losses = AverageMeter()
    error = AverageMeter()

    # Model on train mode
    model.train()

    end = time.time()
    for batch_idx, (input, target) in enumerate(loader):
        # Create vaiables
        if torch.cuda.is_available():
            input_var = torch.autograd.Variable(input.cuda(async=True))
            target_var = torch.autograd.Variable(target.cuda(async=True))
        else:
            input_var = torch.autograd.Variable(input)
            target_var = torch.autograd.Variable(target)

        # compute output
        output = model(input_var)
        loss = torch.nn.functional.cross_entropy(output, target_var)

        # measure accuracy and record loss
        batch_size = target.size(0)
        _, pred = output.data.cpu().topk(1, dim=1)
        error.update(torch.ne(pred.squeeze(), target.cpu()).float().sum() / batch_size, batch_size)
        losses.update(loss.data[0], batch_size)

        # compute gradient and do SGD step
        optimizer.zero_grad()
        loss.backward()
        optimizer.step()

        # measure elapsed time
        batch_time.update(time.time() - end)
        end = time.time()

        # print stats
        if batch_idx % print_freq == 0:
            res = '\t'.join([
                'Epoch: [%d/%d]' % (epoch + 1, n_epochs),
                'Iter: [%d/%d]' % (batch_idx + 1, len(loader)),
                'Time %.3f (%.3f)' % (batch_time.val, batch_time.avg),
                'Loss %.4f (%.4f)' % (losses.val, losses.avg),
                'Error %.4f (%.4f)' % (error.val, error.avg),
            ])
            print(res)

    # Return summary statistics
    return batch_time.avg, losses.avg, error.avg


def test_epoch(model, loader, print_freq=1, is_test=True):
    batch_time = AverageMeter()
    losses = AverageMeter()
    error = AverageMeter()

    # Model on eval mode
    model.eval()

    end = time.time()
    for batch_idx, (input, target) in enumerate(loader):
        # Create vaiables
        if torch.cuda.is_available():
            input_var = torch.autograd.Variable(input.cuda(async=True), volatile=True)
            target_var = torch.autograd.Variable(target.cuda(async=True), volatile=True)
        else:
            input_var = torch.autograd.Variable(input, volatile=True)
            target_var = torch.autograd.Variable(target, volatile=True)

        # compute output
        output = model(input_var)
        loss = torch.nn.functional.cross_entropy(output, target_var)

        # measure accuracy and record loss
        batch_size = target.size(0)
        _, pred = output.data.cpu().topk(1, dim=1)
        error.update(torch.ne(pred.squeeze(), target.cpu()).float().sum() / batch_size, batch_size)
        losses.update(loss.data[0], batch_size)

        # measure elapsed time
        batch_time.update(time.time() - end)
        end = time.time()

        # print stats
#         if batch_idx % print_freq == 0:
#             res = '\t'.join([
#                 'Test' if is_test else 'Valid',
#                 'Iter: [%d/%d]' % (batch_idx + 1, len(loader)),
#                 'Time %.3f (%.3f)' % (batch_time.val, batch_time.avg),
#                 'Loss %.4f (%.4f)' % (losses.val, losses.avg),
#                 'Error %.4f (%.4f)' % (error.val, error.avg),
#             ])
#             print(res)

    # Return summary statistics
    return batch_time.avg, losses.avg, error.avg


def train(model, train_set, test_set, save, n_epochs=300, valid_size=5000,
          batch_size=64, lr=0.1, wd=0.0001, momentum=0.9, seed=None):
    if seed is not None:
        torch.manual_seed(seed)

    # Create train/valid split
    if valid_size:
        indices = torch.randperm(len(train_set))
        train_indices = indices[:len(indices) - valid_size]
        train_sampler = torch.utils.data.sampler.SubsetRandomSampler(train_indices)
        valid_indices = indices[len(indices) - valid_size:]
        valid_sampler = torch.utils.data.sampler.SubsetRandomSampler(valid_indices)

    # Data loaders
    test_loader = torch.utils.data.DataLoader(test_set, batch_size=batch_size, shuffle=False,
                                              pin_memory=(torch.cuda.is_available()), num_workers=0)
    if valid_size:
        train_loader = torch.utils.data.DataLoader(train_set, batch_size=batch_size, sampler=train_sampler,
                                                   pin_memory=(torch.cuda.is_available()), num_workers=0)
        valid_loader = torch.utils.data.DataLoader(train_set, batch_size=batch_size, sampler=valid_sampler,
                                                   pin_memory=(torch.cuda.is_available()), num_workers=0)
    else:
        train_loader = torch.utils.data.DataLoader(train_set, batch_size=batch_size, shuffle=True,
                                                   pin_memory=(torch.cuda.is_available()), num_workers=0)
        valid_loader = None

    # Model on cuda
    if torch.cuda.is_available():
        model = model.cuda()

    # Wrap model for multi-GPUs, if necessary
    model_wrapper = model
    if torch.cuda.is_available() and torch.cuda.device_count() > 1:
        model_wrapper = torch.nn.DataParallel(model).cuda()

    # Optimizer
    optimizer = torch.optim.SGD(model_wrapper.parameters(), lr=lr, momentum=momentum, nesterov=True, weight_decay=wd)
    scheduler = torch.optim.lr_scheduler.MultiStepLR(optimizer, milestones=[0.5 * n_epochs, 0.75 * n_epochs],
                                                     gamma=0.1)

    # Start log
    with open(os.path.join(save, 'results.csv'), 'w') as f:
        f.write('epoch,train_loss,train_error,valid_loss,valid_error,test_error\n')

    # Train model
    best_error = 1
    print_freq = 100
    for epoch in range(n_epochs):
        scheduler.step()
        _, train_loss, train_error = train_epoch(
            model=model_wrapper,
            loader=train_loader,
            optimizer=optimizer,
            epoch=epoch,
            n_epochs=n_epochs,
            print_freq=print_freq,
        )
        _, valid_loss, valid_error = test_epoch(
            model=model_wrapper,
            loader=valid_loader if valid_loader else test_loader,
            is_test=(not valid_loader)
        )

        result_train.append([train_loss, train_error, valid_loss, valid_error])
        print("epoch:", epoch + 1, "/", n_epochs, ", train_loss:", train_loss, ", train_error:", train_error, ", valid_loss:", valid_loss, ",valid_error:", valid_error )
        if epoch > n_epochs-10:
            test_results = test_epoch(
                model=model_wrapper,
                loader=test_loader,
                is_test=True
            )
            _, _, test_error = test_results
            print("epoch:", epoch + 1, "/", n_epochs, ", test_error:", test_error)
            result_test.append(test_results)

        # Determine if model is the best
#         if valid_loader and valid_error < best_error:
#             best_error = valid_error
#             print('New best error: %.4f' % best_error)
#             torch.save(model.state_dict(), os.path.join(save, 'model.dat'))
#         else:
#             torch.save(model.state_dict(), os.path.join(save, 'model.dat'))

        # Log results
        with open(os.path.join(save, 'results.csv'), 'a') as f:
            f.write('%03d,%0.6f,%0.6f,%0.5f,%0.5f,\n' % (
                (epoch + 1),
                train_loss,
                train_error,
                valid_loss,
                valid_error,
            ))
            # f.write('')

        # with open(os.path.join(save, 'results.csv'), 'a') as f:


    # Final test of model on test set
#     model.load_state_dict(torch.load(os.path.join(save, 'model.dat')))
#     if torch.cuda.is_available() and torch.cuda.device_count() > 1:
#         model = torch.nn.DataParallel(model).cuda()
    test_results = test_epoch(
        model=model_wrapper,
        loader=test_loader,
        is_test=True
    )
    _, _, test_error = test_results
    with open(os.path.join(save, 'results.csv'), 'a') as f:
        for results in test_results:
            _, _, test_error = results
            f.write(',,,,,%0.5f\n' % (test_error))
    print('Final test error: %.4f' % test_error)


def count_parameters(model):
    return sum(p.numel() for p in model.parameters() if p.requires_grad)


def demo(data, save, depth=58, growth_rate=12, efficient=False, valid_size=5000,
         n_epochs=80, batch_size=64, seed=None):
    """
    A demo to show off training of efficient DenseNets.
    Trains and evaluates a DenseNet-BC on CIFAR-10.
    Args:
        data (str) - path to directory where data should be loaded from/downloaded
            (default $DATA_DIR)
        save (str) - path to save the model to (default /tmp)
        depth (int) - depth of the network (number of convolution layers) (default 40)
        growth_rate (int) - number of features added per DenseNet layer (default 12)
        efficient (bool) - use the memory efficient implementation? (default True)
        valid_size (int) - size of validation set
        n_epochs (int) - number of epochs for training (default 300)
        batch_size (int) - size of minibatch (default 256)
        seed (int) - manually set the random seed (default None)
    """

    # Get densenet configuration
    if (depth - 4) % 3:
        raise Exception('Invalid depth')
    block_config = [(depth - 4) // 3 for _ in range(3)]

    # Data transforms
    mean = [0.5071, 0.4867, 0.4408]
    stdv = [0.2675, 0.2565, 0.2761]
    train_transforms = transforms.Compose([
        transforms.RandomCrop(32, padding=4),
        transforms.RandomHorizontalFlip(),
        transforms.ToTensor(),
        transforms.Normalize(mean=mean, std=stdv),
    ])
    test_transforms = transforms.Compose([
        transforms.ToTensor(),
        transforms.Normalize(mean=mean, std=stdv),
    ])

    # Datasets

    train_set = datasets.CIFAR10(data, train=True, transform=train_transforms, download=True)
    test_set = datasets.CIFAR10(data, train=False, transform=test_transforms, download=True)

    # Models
    # klass = DenseNetEfficient if efficient else DenseNet
    klass = DenseNet
    model = klass(
        growth_rate=growth_rate,
        block_config=block_config,
        num_classes=10,
        small_inputs=True
    )
    print(model)

    # Make save directory
    if not os.path.exists(save):
        os.makedirs(save)
    if not os.path.isdir(save):
        raise Exception('%s is not a dir' % save)

    print("depth:", depth, "growth_rate:", growth_rate, "batch_size:", batch_size)
    print("n_epochs:", n_epochs, "valid_size:", valid_size, "efficient:", efficient)
    print("count_parameters:", count_parameters(model))

    # Train the model
    train(model=model, train_set=train_set, test_set=test_set, save=save,
          valid_size=valid_size, n_epochs=n_epochs, batch_size=batch_size, seed=seed)
    print("depth:", depth, "growth_rate:", growth_rate, "batch_size:", batch_size)
    print("n_epochs:", n_epochs, "valid_size:", valid_size, "efficient:", efficient)
    print("count_parameters:", count_parameters(model))
    print('Done!')


"""
A demo to show off training of efficient DenseNets.
Trains and evaluates a DenseNet-BC on CIFAR-10.
Try out the efficient DenseNet implementation:
python demo.py --efficient True --data <path_to_data_dir> --save <path_to_save_dir>
Try out the naive DenseNet implementation:
python demo.py --efficient True --data <path_to_data_dir> --save <path_to_save_dir>
Other args:
    --depth (int) - depth of the network (number of convolution layers) (default 40)
    --growth_rate (int) - number of features added per DenseNet layer (default 12)
    --n_epochs (int) - number of epochs for training (default 300)
    --batch_size (int) - size of minibatch (default 256)
    --seed (int) - manually set the random seed (default None)
"""
if __name__ == '__main__':
    # fire.Fire(demo)
    demo("/home/corey/coreys-code/data", "save_58", depth=58, growth_rate=12, efficient=False, valid_size=5000,
         n_epochs=300, batch_size=64, seed=None)
    demo("/home/corey/coreys-code/data", "save_127", depth=127, growth_rate=18, efficient=False, valid_size=5000,
         n_epochs=300, batch_size=64, seed=None)