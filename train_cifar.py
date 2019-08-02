from __future__ import print_function

import argparse
import os
import shutil
import time
import random

import torch
import torch.nn as nn
import torch.nn.parallel
import torch.backends.cudnn as cudnn
import torch.optim as optim
import torch.utils.data as data
import torchvision.transforms as transforms
import torchvision.datasets as datasets
import models.cifar as models

from utils import Logger, AverageMeter, accuracy, mkdir_p, savefig

import sys
srcFolder = os.path.join(os.path.abspath(os.path.dirname(__file__)), 'src')
sys.path.append(srcFolder)
import regressor
import gem


model_names = sorted(name for name in models.__dict__
    if name.islower() and not name.startswith("__")
    and callable(models.__dict__[name]))

parser = argparse.ArgumentParser(description='PyTorch CIFAR10/100 Training')
# Datasets
parser.add_argument('-d', '--dataset', default='cifar10', type=str)
parser.add_argument('-j', '--workers', default=4, type=int, metavar='N',
                    help='number of data loading workers (default: 4)')
# Optimization options
parser.add_argument('--epochs', default=300, type=int, metavar='N',
                    help='number of total epochs to run')
parser.add_argument('--start-epoch', default=0, type=int, metavar='N',
                    help='manual epoch number (useful on restarts)')
parser.add_argument('--train-batch', default=128, type=int, metavar='N',
                    help='train batchsize')
parser.add_argument('--test-batch', default=100, type=int, metavar='N',
                    help='test batchsize')
parser.add_argument('--lr', '--learning-rate', default=0.1, type=float,
                    metavar='LR', help='initial learning rate')
parser.add_argument('--drop', '--dropout', default=0, type=float,
                    metavar='Dropout', help='Dropout ratio')
parser.add_argument('--schedule', type=int, nargs='+', default=[150, 225],
                        help='Decrease learning rate at these epochs.')
parser.add_argument('--gamma', type=float, default=0.1, help='LR is multiplied by gamma on schedule.')
parser.add_argument('--momentum', default=0.9, type=float, metavar='M',
                    help='momentum')
parser.add_argument('--weight-decay', '--wd', default=5e-4, type=float,
                    metavar='W', help='weight decay (default: 1e-4)')
# Checkpoints
parser.add_argument('-c', '--checkpoint', default='checkpoint', type=str, metavar='PATH',
                    help='path to save checkpoint (default: checkpoint)')
parser.add_argument('--resume', default='', type=str, metavar='PATH',
                    help='path to latest checkpoint (default: none)')
# Architecture
parser.add_argument('--arch', '-a', metavar='ARCH', default='resnet20',
                    choices=model_names,
                    help='model architecture: ' +
                        ' | '.join(model_names) +
                        ' (default: resnet18)')
parser.add_argument('--depth', type=int, default=29, help='Model depth.')
parser.add_argument('--cardinality', type=int, default=8, help='Model cardinality (group).')
parser.add_argument('--widen-factor', type=int, default=4, help='Widen factor. 4 -> 64, 8 -> 128, ...')
parser.add_argument('--growthRate', type=int, default=12, help='Growth rate for DenseNet.')
parser.add_argument('--compressionRate', type=int, default=2, help='Compression Rate (theta) for DenseNet.')
# Miscs
parser.add_argument('--manualSeed', type=int, help='manual seed')
parser.add_argument('-e', '--evaluate', dest='evaluate', action='store_true',
                    help='evaluate model on validation set')
#Device options
parser.add_argument('--gpu-id', default='0', type=str,
                    help='id(s) for CUDA_VISIBLE_DEVICES')

parser.add_argument('--datapath', type=str, default='', help='the path to the datasets')
parser.add_argument('--mtype', type=str, default='dcl', help='method type: baseline, dcl, gem')
parser.add_argument('--dcl_refsize', type=int, default=0, help='reference size for DCL or memory size for GEM')
parser.add_argument('--dcl_offset', type=int, default=0, help='offset for reference initialization')
parser.add_argument('--dcl_window', type=int, default=0, help='dcl window for updating accumulated gradient')
parser.add_argument('--gem_memsize', type=int, default=0, help='memory size for GEM')
args = parser.parse_args()
state = {k: v for k, v in args._get_kwargs()}

# Validate dataset
assert args.dataset == 'cifar10' or args.dataset == 'cifar100', 'Dataset can only be cifar10 or cifar100.'

# Use CUDA
#os.environ['CUDA_VISIBLE_DEVICES'] = args.gpu_id
use_cuda = torch.cuda.is_available()

# Random seed
if args.manualSeed is None:
    args.manualSeed = random.randint(1, 10000)
random.seed(args.manualSeed)
torch.manual_seed(args.manualSeed)
if use_cuda:
    torch.cuda.manual_seed_all(args.manualSeed)

best_acc = 0  # best test accuracy

def main():
    global best_acc
    start_epoch = args.start_epoch  # start from epoch 0 or last checkpoint epoch

    if not os.path.isdir(args.checkpoint):
        mkdir_p(args.checkpoint)

    # Data
    print('==> Preparing dataset %s' % args.dataset)
    transform_train = transforms.Compose([
        transforms.RandomCrop(32, padding=4),
        transforms.RandomHorizontalFlip(),
        transforms.ToTensor(),
        transforms.Normalize((0.4914, 0.4822, 0.4465), (0.2023, 0.1994, 0.2010)),
    ])

    transform_test = transforms.Compose([
        transforms.ToTensor(),
        transforms.Normalize((0.4914, 0.4822, 0.4465), (0.2023, 0.1994, 0.2010)),
    ])
    if args.dataset == 'cifar10':
        dataloader = datasets.CIFAR10
        num_classes = 10
    else:
        dataloader = datasets.CIFAR100
        num_classes = 100

    trainset = dataloader(root=args.datapath, train=True, download=False, transform=transform_train)
    trainloader = data.DataLoader(trainset, batch_size=args.train_batch, shuffle=True, num_workers=args.workers)

    testset = dataloader(root=args.datapath, train=False, download=False, transform=transform_test)
    testloader = data.DataLoader(testset, batch_size=args.test_batch, shuffle=False, num_workers=args.workers)

    # Model
    print("==> creating model '{}'".format(args.arch))
    if args.arch.startswith('resnext'):
        model = models.__dict__[args.arch](
                    cardinality=args.cardinality,
                    num_classes=num_classes,
                    depth=args.depth,
                    widen_factor=args.widen_factor,
                    dropRate=args.drop,
                )
    elif args.arch.startswith('densenet'):
        model = models.__dict__[args.arch](
                    num_classes=num_classes,
                    depth=args.depth,
                    growthRate=args.growthRate,
                    compressionRate=args.compressionRate,
                    dropRate=args.drop,
                )
    elif args.arch.startswith('wrn'):
        model = models.__dict__[args.arch](
                    num_classes=num_classes,
                    depth=args.depth,
                    widen_factor=args.widen_factor,
                    dropRate=args.drop,
                )
    elif args.arch.endswith('resnet'):
        model = models.__dict__[args.arch](
                    num_classes=num_classes,
                    depth=args.depth,
                )
    else:
        model = models.__dict__[args.arch](num_classes=num_classes)
    modeltype = type(model).__name__
    model,classifier = decomposeModel(model)
    classifier = classifier.cuda()

    model = torch.nn.DataParallel(model).cuda()
    cudnn.benchmark = True
    print('    Total params: %.2fM' % (sum(p.numel() for p in model.parameters())/1000000.0))

    criterion = nn.CrossEntropyLoss()
    optimizer = optim.SGD(model.parameters(), lr=args.lr, momentum=args.momentum, weight_decay=args.weight_decay)

    reg_net = None
    optimizer_reg = optim.SGD(classifier.parameters(), lr=args.lr, momentum=args.momentum, weight_decay=args.weight_decay)
    if args.mtype == 'baseline':
        args.dcl_refsize = 0
        reg_net = regressor.Net(classifier, optimizer_reg, ref_size=args.dcl_refsize, backendtype=modeltype)
    elif args.mtype == 'dcl':
        reg_net = regressor.Net(classifier, optimizer_reg, ref_size=args.dcl_refsize, backendtype=modeltype, dcl_offset=args.dcl_offset, dcl_window=args.dcl_window)
    elif args.mtype == 'gem':
        reg_net = gem.Net(classifier, optimizer_reg, n_memories=args.gem_memsize, backendtype=modeltype)

    print(args)

    # Resume
    title = 'cifar-10-' + args.arch
    if args.resume:
        # Load checkpoint.
        print('==> Resuming from checkpoint..')
        assert os.path.isfile(args.resume), 'Error: no checkpoint directory found!'
        args.checkpoint = os.path.dirname(args.resume)
        checkpoint = torch.load(args.resume)
        best_acc = checkpoint['best_acc']
        start_epoch = checkpoint['epoch']
        model.load_state_dict(checkpoint['state_dict'])
        optimizer.load_state_dict(checkpoint['optimizer'])
        logger = Logger(os.path.join(args.checkpoint, 'log.txt'), title=title, resume=True)
    else:
        logger = Logger(os.path.join(args.checkpoint, 'log.txt'), title=title)
        logger.set_names(['Learning Rate', 'Train Loss', 'Valid Loss', 'Train Err.', 'Valid Err.', 'Cos Bef.', 'Cos Aft.', 'Mag'])


    if args.evaluate:
        print('\nEvaluation only')
        test_loss, test_acc = test(testloader, model, criterion, start_epoch, use_cuda)
        print(' Test Loss:  %.8f, Test Acc:  %.2f' % (test_loss, test_acc))
        return

    # Train and val
    for epoch in range(start_epoch, args.epochs):
        adjust_learning_rate_two(optimizer, optimizer_reg, epoch)

        print('Epoch: [%d | %d] LR: %f' % (epoch + 1, args.epochs, state['lr']))

        train_loss, train_top1_acc, train_top5_acc, cong_bf, cong_af, mag = train(trainloader, model, criterion, optimizer, epoch, use_cuda, reg_net=reg_net)
        test_loss, test_top1_acc, test_top5_acc = test(testloader, model, criterion, epoch, use_cuda, reg_net=reg_net)
        print('train_loss: {:.4f}, train_top1_err: {:.2f}, train_top5_err: {:.2f}, test_loss: {:.4f}, test_top1_err: {:.2f}, test_top5_err: {:.2f}, cong_bf: {:.4f}, cong_af: {:.4f}, mag: {:.4f}'.format(
            train_loss, 100-train_top1_acc, 100-train_top5_acc,
            test_loss, 100-test_top1_acc, 100-test_top5_acc,
            cong_bf, cong_af, mag))

        # append logger file
        logger.append([state['lr'], train_loss, test_loss, 100-train_top1_acc, 100-test_top1_acc, cong_bf, cong_af, mag])

        # save model
        is_best = test_top1_acc > best_acc
        best_acc = max(test_top1_acc, best_acc)
        save_checkpoint({
                'epoch': epoch + 1,
                'state_dict': model.state_dict(),
                'classifier_state_dict': reg_net.state_dict(),
                'acc': test_top1_acc,
                'best_acc': best_acc,
                'optimizer' : optimizer.state_dict(),
                'cong_bf': cong_bf,
                'cong_af': cong_af,
                'mag': mag,
            }, is_best, checkpoint=args.checkpoint)

    logger.close()

    print('Best err:')
    print(100-best_acc)

def train(trainloader, model, criterion, optimizer, epoch, use_cuda, reg_net=None):
    # switch to train mode
    model.train()

    batch_time = AverageMeter()
    data_time = AverageMeter()
    losses = AverageMeter()
    top1 = AverageMeter()
    top5 = AverageMeter()
    cong_bf = AverageMeter()
    cong_af = AverageMeter()
    mag = AverageMeter()
    end = time.time()

    for batch_idx, (inputs, targets) in enumerate(trainloader):
        if reg_net.__module__ == 'gem' and reg_net.n_memories > 0 and reg_net.mem_cnt <= 0:
            reg_net.setRawMem(inputs, targets)
        # measure data loading time
        data_time.update(time.time() - end)

        if use_cuda:
            inputs, targets = inputs.cuda(), targets.cuda()
        inputs, targets = torch.autograd.Variable(inputs), torch.autograd.Variable(targets)

        if reg_net.__module__ == 'gem' and reg_net.mem_cnt > 0:
            reg_net.genMem(model)

        # compute output
        outputs = model(inputs)
        loss, outputs, cos_sims = reg_net.observe(outputs, targets, optimizer, criterion)

        # measure accuracy and record loss
        prec1, prec5 = accuracy(outputs.data, targets.data, topk=(1, 5))
        losses.update(loss.item(), inputs.size(0))
        top1.update(prec1.item(), inputs.size(0))
        top5.update(prec5.item(), inputs.size(0))
        cong_bf.update(cos_sims[0], inputs.size(0))
        cong_af.update(cos_sims[1], inputs.size(0))
        mag.update(cos_sims[2], inputs.size(0))

        # measure elapsed time
        batch_time.update(time.time() - end)
        end = time.time()
        
    if reg_net.__module__ == 'gem':
            reg_net.mem_cnt = 0
    return (losses.avg, top1.avg, top5.avg, cong_bf.avg, cong_af.avg, mag.avg)

def test(testloader, model, criterion, epoch, use_cuda, reg_net=None):
    global best_acc

    batch_time = AverageMeter()
    data_time = AverageMeter()
    losses = AverageMeter()
    top1 = AverageMeter()
    top5 = AverageMeter()

    # switch to evaluate mode
    model.eval()
    if reg_net is not None:
        reg_net.eval()

    end = time.time()
    for batch_idx, (inputs, targets) in enumerate(testloader):
        # measure data loading time
        data_time.update(time.time() - end)

        if use_cuda:
            inputs, targets = inputs.cuda(), targets.cuda()
        with torch.no_grad():
            inputs, targets = torch.autograd.Variable(inputs), torch.autograd.Variable(targets)
            
            # compute output
            outputs = model(inputs)
            outputs = reg_net(outputs)
            loss = criterion(outputs, targets)

        # measure accuracy and record loss
        prec1, prec5 = accuracy(outputs.data, targets.data, topk=(1, 5))
        losses.update(loss.item(), inputs.size(0))
        top1.update(prec1.item(), inputs.size(0))
        top5.update(prec5.item(), inputs.size(0))

        # measure elapsed time
        batch_time.update(time.time() - end)
        end = time.time()
      
    return (losses.avg, top1.avg, top5.avg)

def save_checkpoint(state, is_best, checkpoint='checkpoint', filename='checkpoint.pth.tar'):
    filepath = os.path.join(checkpoint, filename)
    torch.save(state, filepath)
    if is_best:
        shutil.copyfile(filepath, os.path.join(checkpoint, 'model_best.pth.tar'))

def adjust_learning_rate(optimizer, epoch):
    global state
    if epoch in args.schedule:
        state['lr'] *= args.gamma
        for param_group in optimizer.param_groups:
            param_group['lr'] = state['lr']

def adjust_learning_rate_two(optimizer, optimizer1, epoch):
    global state
    if epoch in args.schedule:
        state['lr'] *= args.gamma
        for param_group in optimizer.param_groups:
            param_group['lr'] = state['lr']
        for param_group in optimizer1.param_groups:
            param_group['lr'] = state['lr']

def decomposeModel(model):
    model_part1 = None
    model_part2 = None
    if type(model).__name__=='CifarResNeXt':
        tempList = list(model.children())
        model_part1 = nn.Sequential(*tempList[:-1], nn.AvgPool2d(kernel_size=8, stride=1))
        model_part2 = tempList[-1]
    elif type(model).__name__=='DenseNet':
        tempList = list(model.children())
        model_part1 = nn.Sequential(*tempList[:-1])
        model_part2 = tempList[-1]
    elif type(model).__name__=='WideResNet':
        tempList = list(model.children())
        model_part1 = nn.Sequential(*tempList[:-1])
        model_part2 = tempList[-1]
    else:
        tempList = list(model.children())
        model_part1 = nn.Sequential(*tempList[:-1])
        model_part2 = tempList[-1]
    return (model_part1, model_part2)

if __name__ == '__main__':
    main()
