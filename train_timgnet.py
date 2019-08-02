import argparse
import os
import shutil
import time
import random
import numpy as np

import torch
import torch.nn as nn
import torch.nn.parallel
import torch.backends.cudnn as cudnn
import torch.optim as optim
import torch.utils.data as data
import torchvision.transforms as transforms
import torchvision.datasets as datasets
import torchvision.models as models
import models.imagenet as customized_models

# uncomment the following line to use tensorboard
# from tensorboardX import SummaryWriter

from PIL import Image

from utils import Logger, AverageMeter, accuracy, mkdir_p, savefig

import sys
srcFolder = os.path.join(os.path.abspath(os.path.dirname(__file__)), 'src')
sys.path.append(srcFolder)
import regressor
import gem

def parseClasses(file):
    classes = []
    filenames = []
    with open(file) as f:
        lines = f.readlines()
    lines = [x.strip() for x in lines]
    for x in range(0,len(lines)):
        tokens = lines[x].split()
        classes.append(tokens[1])
        filenames.append(tokens[0])
    return filenames,classes

class TImgNetDataset(data.Dataset):
    """Dataset wrapping images and ground truths."""
    
    def __init__(self, img_path, gt_path, class_to_idx=None, transform=None):
        self.img_path = img_path
        self.transform = transform
        self.gt_path = gt_path
        self.class_to_idx = class_to_idx
        self.classidx = []
        self.imgs, self.classnames = parseClasses(gt_path)
        for classname in self.classnames:
            self.classidx.append(self.class_to_idx[classname])

    def __getitem__(self, index):
            """
            Args:
                index (int): Index
            Returns:
                tuple: (image, y) where y is the label of the image.
            """
            img = None
            with open(os.path.join(self.img_path, self.imgs[index]), 'rb') as f:
                img = Image.open(f)
                img = img.convert('RGB')
                if self.transform is not None:
                    img = self.transform(img)
            y = self.classidx[index]
            return img, y

    def __len__(self):
        return len(self.imgs)

# Models
default_model_names = sorted(name for name in models.__dict__
    if name.islower() and not name.startswith("__")
    and callable(models.__dict__[name]))

customized_models_names = sorted(name for name in customized_models.__dict__
    if name.islower() and not name.startswith("__")
    and callable(customized_models.__dict__[name]))

for name in customized_models.__dict__:
    if name.islower() and not name.startswith("__") and callable(customized_models.__dict__[name]):
        models.__dict__[name] = customized_models.__dict__[name]

model_names = default_model_names + customized_models_names

# Parse arguments
parser = argparse.ArgumentParser(description='PyTorch ImageNet Training')

# Datasets
parser.add_argument('-d', '--data', default='path to dataset', type=str)
parser.add_argument('-j', '--workers', default=4, type=int, metavar='N',
                    help='number of data loading workers (default: 4)')
# Optimization options
parser.add_argument('--epochs', default=90, type=int, metavar='N',
                    help='number of total epochs to run')
parser.add_argument('--start-epoch', default=0, type=int, metavar='N',
                    help='manual epoch number (useful on restarts)')
parser.add_argument('--train-batch', default=256, type=int, metavar='N',
                    help='train batchsize (default: 256)')
parser.add_argument('--test-batch', default=200, type=int, metavar='N',
                    help='test batchsize (default: 200)')
parser.add_argument('--lr', '--learning-rate', default=0.1, type=float,
                    metavar='LR', help='initial learning rate')
parser.add_argument('--drop', '--dropout', default=0, type=float,
                    metavar='Dropout', help='Dropout ratio')
parser.add_argument('--schedule', type=int, nargs='+', default=[150, 225],
                        help='Decrease learning rate at these epochs.')
parser.add_argument('--gamma', type=float, default=0.1, help='LR is multiplied by gamma on schedule.')
parser.add_argument('--momentum', default=0.9, type=float, metavar='M',
                    help='momentum')
parser.add_argument('--weight-decay', '--wd', default=1e-4, type=float,
                    metavar='W', help='weight decay (default: 1e-4)')
# Checkpoints
parser.add_argument('-c', '--checkpoint', default='checkpoint', type=str, metavar='PATH',
                    help='path to save checkpoint (default: checkpoint)')
parser.add_argument('--resume', default='', type=str, metavar='PATH',
                    help='path to latest checkpoint (default: none)')
# Architecture
parser.add_argument('--arch', '-a', metavar='ARCH', default='resnet18',
                    choices=model_names,
                    help='model architecture: ' +
                        ' | '.join(model_names) +
                        ' (default: resnet18)')
parser.add_argument('--depth', type=int, default=29, help='Model depth.')
parser.add_argument('--cardinality', type=int, default=32, help='ResNet cardinality (group).')
parser.add_argument('--base-width', type=int, default=4, help='ResNet base width.')
parser.add_argument('--widen-factor', type=int, default=4, help='Widen factor. 4 -> 64, 8 -> 128, ...')
# Miscs
parser.add_argument('--manualSeed', type=int, help='manual seed')
parser.add_argument('-e', '--evaluate', dest='evaluate', action='store_true',
                    help='evaluate model on validation set')
parser.add_argument('--pretrained', dest='pretrained', action='store_true',
                    help='use pre-trained model')
#Device options
parser.add_argument('--gpu-id', default='0', type=str,
                    help='id(s) for CUDA_VISIBLE_DEVICES')
parser.add_argument('--n_classes', type=int, default=1000, help='num of classes')
parser.add_argument('--pretrained_model', type=str, default='', help='pre-trained model')
parser.add_argument('--use_pretrained', type=int, default=1, help='use the pretrained model or not')
parser.add_argument('--mtype', type=str, default='dcl', help='method type: baseline, dcl, gem')
parser.add_argument('--dcl_refsize', type=int, default=0, help='reference size for DCL or memory size for GEM')
parser.add_argument('--dcl_offset', type=int, default=0, help='offset for reference initialization')
parser.add_argument('--dcl_window', type=int, default=0, help='dcl window for updating accumulated gradient')
parser.add_argument('--gem_memsize', type=int, default=0, help='memory size for GEM')
parser.add_argument('--save_err_at_epoch', type=str, default='', help='path to save errors at each epoch, if empty then nothing will be saved')
args = parser.parse_args()
state = {k: v for k, v in args._get_kwargs()}

# Use CUDA
use_cuda = torch.cuda.is_available()

# Random seed
if args.manualSeed is None:
    args.manualSeed = random.randint(1, 10000)
random.seed(args.manualSeed)
torch.manual_seed(args.manualSeed)
if use_cuda:
    torch.cuda.manual_seed_all(args.manualSeed)

best_acc = 0  # best test accuracy

modelzoo = {
    'densenet169': models.densenet169,
    'vgg16': models.vgg16,
    'resnet101': models.resnet101,
    'resnet50': models.resnet50,
    'resnet34': models.resnet34,
    'resnet18': models.resnet18,
}

def main():
    global best_acc
    start_epoch = args.start_epoch  # start from epoch 0 or last checkpoint epoch

    if not os.path.isdir(args.checkpoint):
        mkdir_p(args.checkpoint)

    # uncomment the following line to use tensorboard
    # writer = SummaryWriter(os.path.join(args.checkpoint, 'logs'))

    # Data loading
    traindir = os.path.join(args.data, 'train')
    valdir = os.path.join(args.data, 'val')
    normalize = transforms.Normalize(mean=[0.485, 0.456, 0.406],
                                     std=[0.229, 0.224, 0.225])

    train_loader = torch.utils.data.DataLoader(
        datasets.ImageFolder(traindir, transforms.Compose([
            transforms.Resize(224),
            transforms.RandomHorizontalFlip(),
            transforms.ToTensor(),
            normalize,
        ])),
        batch_size=args.train_batch, shuffle=True,
        num_workers=args.workers, pin_memory=True)
    
    valdir = os.path.join(args.data, 'val', 'images')
    valgtfile = os.path.join(args.data, 'val', 'val_annotations.txt')
    val_dataset = TImgNetDataset(valdir, valgtfile, class_to_idx=train_loader.dataset.class_to_idx.copy(),
            transform=transforms.Compose([
            transforms.Resize(224),
            transforms.ToTensor(),
            normalize,
            ]))
    val_loader = torch.utils.data.DataLoader(
        val_dataset,
        batch_size=args.test_batch, shuffle=False,
        num_workers=args.workers, pin_memory=True)

    # create model
    if args.pretrained_model:
        print("=> using pre-trained model '{}'".format(args.pretrained_model))
        model = modelzoo[args.pretrained_model](pretrained=bool(args.use_pretrained))
    elif args.arch.startswith('resnext'):
        model = models.__dict__[args.arch](
                    baseWidth=args.base_width,
                    cardinality=args.cardinality,
                )
    else:
        print("=> creating model '{}'".format(args.arch))
        model = models.__dict__[args.arch]()
    modeltype = type(model).__name__
    # split a model into two parts, i.e., a DNN and a FC layer
    model,classifier = decomposeModel(model, args.n_classes)
    classifier = classifier.cuda()

    if args.arch.startswith('alexnet') or args.arch.startswith('vgg'):
        model.features = torch.nn.DataParallel(model.features)
        model.cuda()
    else:
        model = torch.nn.DataParallel(model).cuda()

    cudnn.benchmark = True
    print('    Total params: %.2fM' % (sum(p.numel() for p in model.parameters())/1000000.0))

    # define loss function (criterion) and optimizer
    criterion = nn.CrossEntropyLoss().cuda()
    optimizer = optim.SGD(model.parameters(), lr=args.lr, momentum=args.momentum, weight_decay=args.weight_decay)

    # define a regressor to wrap the classifier and its optimizer for gradient modification
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
    title = 'Tiny ImageNet-' + args.arch
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
        logger = Logger(os.path.join(args.checkpoint, 'stat.csv'), title=title, resume=True)
    else:
        logger = Logger(os.path.join(args.checkpoint, 'stat.csv'), title=title)
        # Cong_Bef stands for congruency before gradient modification
        # Cong_Aft stands for congruency after gradient modification
        # Magn stands for magnitude
        logger.set_names(['Epoch', 'Tr_T1Acc', 'Val_T1Acc', 'Val_T5Acc', 
                                'Cong_Bef', 'Cong_Aft', 'Magn', 'LR', 'Tr_loss', 'Val_loss',
                                'Tr_batch_time', 'Tr_data_time', 'Val_batch_time', 'Val_data_time'])


    if args.evaluate:
        print('\nEvaluation only')
        test_loss, test_acc = test(val_loader, model, criterion, start_epoch, use_cuda)
        print(' Test Loss:  %.8f, Test Acc:  %.2f' % (test_loss, test_acc))
        return

    # Train and val
    for epoch in range(start_epoch, args.epochs):
        #adjust_learning_rate for two optimizers, i.e., one for the DNN trunk while the other for the regressor
        adjust_learning_rate_two(optimizer, optimizer_reg, epoch)

        print('Epoch: [%d | %d] LR: %f' % (epoch + 1, args.epochs, state['lr']))

        train_loss, train_top1_acc, train_top5_acc, cong_bf, cong_af, mag, tr_batch_time, tr_data_time = train(train_loader, model, criterion, optimizer, epoch, use_cuda, reg_net=reg_net)
        test_loss, test_top1_acc, test_top5_acc, te_batch_time, te_data_time = test(val_loader, model, criterion, epoch, use_cuda, reg_net=reg_net)
        print('train_loss: {:.4f}, train_top1_err: {:.2f}, train_top5_err: {:.2f}, test_loss: {:.4f}, test_top1_err: {:.2f}, test_top5_err: {:.2f}, cong_bf: {:.4f}, cong_af: {:.4f}, mag: {:.4f}'.format(
            train_loss, 100-train_top1_acc, 100-train_top5_acc,
            test_loss, 100-test_top1_acc, 100-test_top5_acc,
            cong_bf, cong_af, mag))

        # uncomment the following line to use tensorboard
        # writer.add_scalars('top-1 err', {'train top-1': 100-train_top1_acc, 'val top-1': 100-test_top1_acc}, epoch+1)
        logger.append([epoch + 1, 100-train_top1_acc, 100-test_top1_acc, 100-test_top5_acc, cong_bf, cong_af, mag, 
                        state['lr'], train_loss, test_loss,
                        tr_batch_time, tr_data_time, te_batch_time, te_data_time])

        # save model
        is_best = test_top1_acc > best_acc
        best_acc = max(test_top1_acc, best_acc)
        save_checkpoint({
                'epoch': epoch + 1,
                'state_dict': model.state_dict(),
                'classifier_state_dict': reg_net.state_dict(),
                'err': 100-test_top1_acc,
                'best_acc': best_acc,
                'optimizer' : optimizer.state_dict(),
                'cong_bf': cong_bf,
                'cong_af': cong_af,
                'mag': mag,
            }, is_best, checkpoint=args.checkpoint)

    logger.close()
    # logger.plot()
    # savefig(os.path.join(args.checkpoint, 'log.eps'))

    print('Best err:')
    print(100-best_acc)

def train(train_loader, model, criterion, optimizer, epoch, 
            use_cuda, reg_net=None, save_err_at_epoch=''):
    # switch to train mode
    model.train()
    reg_net.train()

    batch_time = AverageMeter()
    data_time = AverageMeter()
    losses = AverageMeter()
    top1 = AverageMeter()
    top5 = AverageMeter()
    cong_bf = AverageMeter()
    cong_af = AverageMeter()
    mag = AverageMeter()
    end = time.time()

    err_vs_step = []
    
    for batch_idx, (inputs, targets) in enumerate(train_loader):
        # measure data loading time
        data_time.update(time.time() - end)

        if reg_net.__module__ == 'gem' and reg_net.n_memories > 0 and reg_net.mem_cnt <= 0:
            reg_net.setRawMem(inputs, targets)

        if use_cuda:
            inputs, targets = inputs.cuda(), targets.cuda()
        inputs, targets = torch.autograd.Variable(inputs), torch.autograd.Variable(targets)

        if reg_net.__module__ == 'gem' and reg_net.mem_cnt > 0:
            reg_net.genMem(model)

        # compute output features
        outputs = model(inputs)
        # compute predictions and modify gradient if needed
        loss, outputs, cos_sims = reg_net.observe(outputs, targets, optimizer, criterion)

        # measure accuracy and record loss
        prec1, prec5 = accuracy(outputs.data, targets.data, topk=(1, 5))
        losses.update(loss.item(), inputs.size(0))
        top1.update(prec1.item(), inputs.size(0))
        top5.update(prec5.item(), inputs.size(0))
        cong_bf.update(cos_sims[0], inputs.size(0))
        cong_af.update(cos_sims[1], inputs.size(0))
        mag.update(cos_sims[2], inputs.size(0))

        if save_err_at_epoch != '':
            err_vs_step.append(100-prec1.item())

        # measure elapsed time
        batch_time.update(time.time() - end)
        end = time.time()

    if save_err_at_epoch != '':
        np.savez(os.path.join(save_err_at_epoch, 'err_ep{:02d}.npz'.format(epoch+1)), err=err_vs_step)
    if reg_net.__module__ == 'gem':
            reg_net.mem_cnt = 0
    return (losses.avg, top1.avg, top5.avg, cong_bf.avg, cong_af.avg, mag.avg, batch_time.avg, data_time.avg)

def test(val_loader, model, criterion, epoch, use_cuda, reg_net=None):
    global best_acc

    batch_time = AverageMeter()
    data_time = AverageMeter()
    losses = AverageMeter()
    top1 = AverageMeter()
    top5 = AverageMeter()

    # switch to evaluate mode
    model.eval()
    reg_net.eval()

    end = time.time()
    for batch_idx, (inputs, targets) in enumerate(val_loader):
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

    return (losses.avg, top1.avg, top5.avg, batch_time.avg, data_time.avg)

def evaluate(val_loader, model, criterion, epoch, use_cuda, reg_net=None):
    global best_acc

    batch_time = AverageMeter()
    data_time = AverageMeter()
    losses = AverageMeter()
    top1 = AverageMeter()
    top5 = AverageMeter()

    # switch to evaluate mode
    model.eval()
    reg_net.eval()

    end = time.time()
    for batch_idx, (inputs, targets) in enumerate(val_loader):
        # measure data loading time
        data_time.update(time.time() - end)

        if use_cuda:
            inputs, targets = inputs.cuda(), targets.cuda()
        with torch.no_grad():
            inputs, targets = torch.autograd.Variable(inputs), torch.autograd.Variable(targets)

            # compute predictions
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

def decomposeModel(model, n_classes):
    model_part1 = None
    model_part2 = None
    if type(model).__name__=='CifarResNeXt':
        tempList = list(model.children())
        model_part1 = nn.Sequential(*tempList[:-1])
        model_part2 = tempList[-1]
    elif type(model).__name__=='DenseNet':
        tempList = list(model.children())
        model_part1 = nn.Sequential(*tempList[:-1], nn.ReLU(inplace=True), 
                            nn.MaxPool2d(kernel_size=7, stride=1))
        model_part2 = tempList[-1]
        if tempList[-1].out_features != n_classes:
            model_part2 = nn.Linear(in_features=tempList[-1].in_features, 
                out_features=n_classes)
    elif type(model).__name__=='WideResNet':
        tempList = list(model.children())
        model_part1 = nn.Sequential(*tempList[:-1])
        model_part2 = tempList[-1]
    elif type(model).__name__=='ResNet':
        tempList = list(model.children())
        pooling = nn.MaxPool2d(kernel_size=tempList[-2].kernel_size,
            stride=tempList[-2].stride, 
            padding=tempList[-2].padding, 
            ceil_mode=tempList[-2].ceil_mode)
        model_part1 = nn.Sequential(*tempList[:-2],pooling)
        model_part2 = tempList[-1]
        if tempList[-1].out_features != n_classes:
            model_part2 = nn.Linear(in_features=tempList[-1].in_features, 
                out_features=n_classes)
    else:
        tempList = list(model.children())
        model_part1 = nn.Sequential(*tempList[:-1])
        model_part2 = tempList[-1]
    return (model_part1, model_part2)

if __name__ == '__main__':
    main()
