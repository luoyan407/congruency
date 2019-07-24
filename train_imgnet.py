import argparse
import os
import shutil
import time

import torch
import torch.nn as nn
import torch.nn.parallel
import torch.backends.cudnn as cudnn
import torch.distributed as dist
import torch.optim
import torch.utils.data
import torch.utils.data.distributed
import torchvision.transforms as transforms
import torchvision.datasets as datasets
import torchvision.models as models

# uncomment the following line to use tensorboard
#from tensorboardX import SummaryWriter

from utils import Logger, AverageMeter, accuracy, mkdir_p, savefig
from src.utils import *

import sys
srcFolder = os.path.join(os.path.abspath(os.path.dirname(__file__)), 'src')
sys.path.append(srcFolder)
import regressor
import gem

model_names = sorted(name for name in models.__dict__
    if name.islower() and not name.startswith("__")
    and callable(models.__dict__[name]))

parser = argparse.ArgumentParser(description='PyTorch ImageNet Training')
parser.add_argument('data', metavar='DIR',
                    help='path to dataset')
parser.add_argument('--arch', '-a', metavar='ARCH', default='resnet18',
                    choices=model_names,
                    help='model architecture: ' +
                        ' | '.join(model_names) +
                        ' (default: resnet18)')
parser.add_argument('-j', '--workers', default=4, type=int, metavar='N',
                    help='number of data loading workers (default: 4)')
parser.add_argument('--epochs', default=90, type=int, metavar='N',
                    help='number of total epochs to run')
parser.add_argument('--start-epoch', default=0, type=int, metavar='N',
                    help='manual epoch number (useful on restarts)')
parser.add_argument('-b', '--batch-size', default=256, type=int,
                    metavar='N', help='mini-batch size (default: 256)')
parser.add_argument('--lr', '--learning-rate', default=0.1, type=float,
                    metavar='LR', help='initial learning rate')
parser.add_argument('--momentum', default=0.9, type=float, metavar='M',
                    help='momentum')
parser.add_argument('--weight-decay', '--wd', default=1e-4, type=float,
                    metavar='W', help='weight decay (default: 1e-4)')
parser.add_argument('--print-freq', '-p', default=10, type=int,
                    metavar='N', help='print frequency (default: 10)')
parser.add_argument('--resume', default='', type=str, metavar='PATH',
                    help='path to latest checkpoint (default: none)')
parser.add_argument('-e', '--evaluate', dest='evaluate', action='store_true',
                    help='evaluate model on validation set')
parser.add_argument('--pretrained', dest='pretrained', action='store_true',
                    help='use pre-trained model')
parser.add_argument('--world-size', default=1, type=int,
                    help='number of distributed processes')
parser.add_argument('--dist-url', default='tcp://224.66.41.62:23456', type=str,
                    help='url used to set up distributed training')
parser.add_argument('--dist-backend', default='gloo', type=str,
                    help='distributed backend')
parser.add_argument('--checkpoint-path', default='', type=str, metavar='PATH',
                    help='path to latest checkpoint (default: none)')

parser.add_argument('--lr-decay-epoch', default=30, type=float,
                    help='learning rate decay epoch')
parser.add_argument('--n-classes', default=1000, type=int, metavar='N',
                    help='number of categories')
parser.add_argument('--mtype', type=str, default='dcl', 
                    help='Method type: baseline, dcl, gem')
parser.add_argument('--buffer-size', type=int, default=0, 
                    help='number of accumulated gradients for DCL')
parser.add_argument('--dcl-refsize', type=int, default=0, 
                    help='number of accumulated gradients for DCL')
parser.add_argument('--dcl-window', type=int, default=0, help='Memory window')
parser.add_argument('--dcl-offset', type=int, default=0, help='Memory offset')
parser.add_argument('--dcl-knlg-decay', type=float, default=0.0, help='knowledge decay for DCL')
parser.add_argument('--dcl-QP-margin', type=float, default=0.5, help='QP margin for DCL')
parser.add_argument('--gem-memsize', type=int, default=0, help='memory size for GEM')

best_prec1 = 0


def main():
    global args, best_prec1
    args = parser.parse_args()

    args.distributed = args.world_size > 1
    
    if not os.path.exists(args.checkpoint_path):
        os.makedirs(args.checkpoint_path)

    # uncomment the following line to use tensorboard
    #writer = SummaryWriter(os.path.join(args.checkpoint_path, 'logs'))

    if args.distributed:
        dist.init_process_group(backend=args.dist_backend, init_method=args.dist_url,
                                world_size=args.world_size)

    print(args)

    # create model
    if args.pretrained:
        print("=> using pre-trained model '{}'".format(args.arch))
        model = models.__dict__[args.arch](pretrained=True)
    else:
        print("=> creating model '{}'".format(args.arch))
        model = models.__dict__[args.arch]()

    # Currently does not support training on a distributed system
    backend_name = type(model).__name__
    # split a model into two parts, i.e., a DNN and a FC layer
    model,classifier = decomposeModel(model, args.n_classes, keep_pre_pooling=True)
    classifier = classifier.cuda()
    model = torch.nn.DataParallel(model).cuda()

    # define loss function (criterion) and optimizer
    criterion = nn.CrossEntropyLoss().cuda()

    optimizer = torch.optim.SGD(model.parameters(), args.lr,
                                momentum=args.momentum,
                                weight_decay=args.weight_decay)

    # define a regressor to wrap the classifier and its optimizer for gradient modification
    reg_net = None
    optimizer_reg = optim.SGD(classifier.parameters(), lr=args.lr, momentum=args.momentum, weight_decay=args.weight_decay)
    if args.mtype == 'baseline':
        args.dcl_refsize = 0
        reg_net = regressor.Net(classifier, optimizer_reg, 
                                ref_size=args.dcl_refsize, 
                                backendtype=backend_name)
    elif args.mtype == 'dcl':
        reg_net = regressor.Net(classifier, optimizer_reg, 
                                ref_size=args.dcl_refsize, 
                                backendtype=backend_name, 
                                dcl_offset=args.dcl_offset, 
                                dcl_window=args.dcl_window, 
                                QP_margin=args.dcl_QP_margin)
    elif args.mtype == 'gem':
        reg_net = gem.Net(classifier, optimizer_reg, 
                            n_memories=args.gem_memsize, 
                            backendtype=backend_name)

    # optionally resume from a checkpoint
    if args.resume:
        if os.path.isfile(args.resume):
            print("=> loading checkpoint '{}'".format(args.resume))
            checkpoint = torch.load(args.resume)
            args.start_epoch = checkpoint['epoch']
            best_prec1 = checkpoint['best_prec1']
            model.load_state_dict(checkpoint['state_dict'])
            optimizer.load_state_dict(checkpoint['optimizer'])
            # --------
            optimizer_reg.load_state_dict(checkpoint['optimizer_reg'])
            reg_net.load_state_dict(checkpoint['reg_net'])
            reg_net.opt = optimizer_reg
            reg_net.ref_size = checkpoint['reg_net_ref_size']
            reg_net.backendtype = checkpoint['reg_net_backendtype']
            reg_net.dcl_window = checkpoint['reg_net_dcl_window']
            reg_net.ref_cnt = checkpoint['reg_net_dcl_ref_cnt']
            reg_net.ref_data = checkpoint['reg_net_dcl_accum_grad'].cuda() if checkpoint['reg_net_dcl_accum_grad'] is not None else None
            reg_net.stat_w1 = checkpoint['stat_ref_weight'].cuda()
            print("=> loaded checkpoint '{}' (epoch {})"
                  .format(args.resume, checkpoint['epoch']))
        else:
            print("=> no checkpoint found at '{}'".format(args.resume))
        shutil.copyfile(os.path.join(args.checkpoint_path, 'stat.csv'),
                        os.path.join(args.checkpoint_path, 'stat_end_ep{}.csv'.format(args.start_epoch)))
        logger = Logger(os.path.join(args.checkpoint_path, 'stat.csv'), title='ImageNet', resume=True)
    else:
        logger = Logger(os.path.join(args.checkpoint_path, 'stat.csv'), title='ImageNet')
        logger.set_names(['Epoch', 'Tr_T1Acc', 'Val_T1Acc', 'Val_T5Acc', 
                                'Cong_Bef', 'Cong_Aft', 'Magn', 'LR', 'Tr_loss', 'Val_loss',
                                'Tr_batch_time', 'Tr_data_time', 'Val_batch_time', 'Val_data_time'])

    cudnn.benchmark = True

    # setup data preprocessing procedure
    traindir = os.path.join(args.data, 'train')
    valdir = os.path.join(args.data, 'val')
    normalize = transforms.Normalize(mean=[0.485, 0.456, 0.406],
                                     std=[0.229, 0.224, 0.225])

    train_dataset = datasets.ImageFolder(
        traindir,
        transforms.Compose([
            transforms.RandomResizedCrop(224),
            transforms.RandomHorizontalFlip(),
            transforms.ToTensor(),
            normalize,
        ]))

    if args.distributed:
        train_sampler = torch.utils.data.distributed.DistributedSampler(train_dataset)
    else:
        train_sampler = None

    # load training data
    train_loader = torch.utils.data.DataLoader(
        train_dataset, batch_size=args.batch_size, shuffle=(train_sampler is None),
        num_workers=args.workers, pin_memory=True, sampler=train_sampler)
    # load validation data
    val_loader = torch.utils.data.DataLoader(
        datasets.ImageFolder(valdir, transforms.Compose([
            transforms.Resize(256),
            transforms.CenterCrop(224),
            transforms.ToTensor(),
            normalize,
        ])),
        batch_size=args.batch_size, shuffle=False,
        num_workers=args.workers, pin_memory=True)

    if args.evaluate:
        validate(val_loader, model, criterion)
        return

    scores =  []
    for epoch in range(args.start_epoch, args.epochs):
        if args.distributed:
            train_sampler.set_epoch(epoch)
        cur_lr = adjust_learning_rate_multiopt(optimizer, optimizer_reg, epoch, lr_decay_epoch=args.lr_decay_epoch)

        # train for one epoch
        tr_loss, tr_top1, tr_top5, tr_cong_bf, tr_cong_af, tr_magn, tr_batch_time, tr_data_time = train(train_loader, model, criterion, optimizer, epoch, reg_net=reg_net)

        # evaluate on validation set
        val_loss, prec1, prec5, val_batch_time, val_data_time = validate(val_loader, model, criterion, reg_net=reg_net)
        scores.append(prec1)
        print('--max top-1 acc: {}'.format(max(scores)))

        # uncomment the following lines to use tensorboard
        #writer.add_scalars('top-1 acc', {'train top-1': tr_top1, 'val top-1': prec1}, epoch+1)
        #writer.add_scalars('congruency', {'tr_cong_bf': tr_cong_bf, 'tr_cong_af': tr_cong_af}, epoch+1)
        #writer.add_scalars('loss', {'train loss': tr_loss, 'val loss': val_loss}, epoch+1)
        logger.append([epoch+1, tr_top1, prec1, prec5, 
                        tr_cong_bf, tr_cong_af, tr_magn, cur_lr, tr_loss, val_loss, 
                        tr_batch_time, tr_data_time, val_batch_time, val_data_time])

        # remember best prec@1 and save checkpoint
        is_best = prec1 > best_prec1
        best_prec1 = max(prec1, best_prec1)
        save_checkpoint({
            'epoch': epoch + 1,
            'arch': args.arch,
            'state_dict': model.state_dict(),
            'best_prec1': best_prec1,
            'optimizer' : optimizer.state_dict(),
            'optimizer_reg' : optimizer_reg.state_dict(),
            'reg_net' : reg_net.state_dict(),
            'reg_net_ref_size': reg_net.ref_size if args.mtype != 'gem' else 0,
            'reg_net_backendtype': reg_net.backendtype,
            'reg_net_dcl_window': reg_net.dcl_window if args.mtype != 'gem' else 0,
            'reg_net_dcl_ref_cnt': reg_net.ref_cnt if args.mtype != 'gem' else 0,
            'reg_net_dcl_accum_grad': reg_net.ref_data.clone().cpu() if args.mtype != 'gem' and reg_net.ref_data is not None else None,
            'stat_ref_weight': reg_net.stat_w1.clone().cpu(),
        }, is_best, save_dir=args.checkpoint_path)
    logger.close()


def train(train_loader, model, criterion, optimizer, epoch, reg_net=None):
    batch_time = AverageMeter()
    data_time = AverageMeter()
    losses = AverageMeter()
    top1 = AverageMeter()
    top5 = AverageMeter()

    cong_bf = AverageMeter()
    cong_af = AverageMeter()
    mag = AverageMeter()

    # switch to train mode
    model.train()
    reg_net.train()

    end = time.time()
    for i, (input, target) in enumerate(train_loader):
        # measure data loading time
        data_time.update(time.time() - end)

        if reg_net.__module__ == 'gem' and reg_net.n_memories > 0 and reg_net.mem_cnt <= 0:
            reg_net.setRawMem(input, target)

        target = target.cuda()

        if reg_net.__module__ == 'gem' and reg_net.mem_cnt > 0:
            reg_net.genMem(model)

        # compute output features
        output = model(input)
        # yield predictions
        loss, output, congs = reg_net.observe(output, target, optimizer, criterion)

        # measure accuracy and record loss
        prec1, prec5 = accuracy(output, target, topk=(1, 5))
        losses.update(loss.item(), input.size(0))
        top1.update(prec1[0], input.size(0))
        top5.update(prec5[0], input.size(0))

        cong_bf.update(tensorOrFloat2Float(congs[0]), input.size(0))
        cong_af.update(tensorOrFloat2Float(congs[1]), input.size(0))
        mag.update(tensorOrFloat2Float(congs[2]), input.size(0))

        # measure elapsed time
        batch_time.update(time.time() - end)
        end = time.time()
    
    # Cong_Bef stands for congruency before gradient modification
    # Cong_Aft stands for congruency after gradient modification
    # Magn stands for magnitude
    print('Epoch: [{0}]\t'
          'Time {batch_time.avg:.3f}\t'
          'Data {data_time.avg:.3f}\t'
          'Loss {loss.avg:.4f}\t'
          'Prec@1 {top1.avg:.4f}\t'
          'Prec@5 {top5.avg:.4f}\t'
          'Cong_BF {cong_bf:.4f}\t'
          'Cong_AF {cong_af:.4f}\t'
          'Magn {magn:.4f}'.format(
           epoch, batch_time=batch_time,
           data_time=data_time, loss=losses, top1=top1, top5=top5,
           cong_bf=cong_bf.avg,cong_af=cong_af.avg,magn=mag.avg))

    return losses.avg, top1.avg, top5.avg, cong_bf.avg, cong_af.avg, mag.avg, batch_time.avg, data_time.avg


def validate(val_loader, model, criterion, reg_net=None):
    batch_time = AverageMeter()
    data_time = AverageMeter()
    losses = AverageMeter()
    top1 = AverageMeter()
    top5 = AverageMeter()

    # switch to evaluate mode
    model.eval()
    reg_net.eval()

    with torch.no_grad():
        end = time.time()
        for i, (input, target) in enumerate(val_loader):
            data_time.update(time.time() - end)

            target = target.cuda()

            # compute output and loss
            output = model(input)
            output = reg_net(output)
            loss = criterion(output, target)

            # measure accuracy and record loss
            prec1, prec5 = accuracy(output, target, topk=(1, 5))
            losses.update(loss.item(), input.size(0))
            top1.update(prec1[0], input.size(0))
            top5.update(prec5[0], input.size(0))

            # measure elapsed time
            batch_time.update(time.time() - end)
            end = time.time()
            
        print(' * Prec@1 {top1.avg:.3f} Prec@5 {top5.avg:.3f}'
              .format(top1=top1, top5=top5))

    return losses.avg, top1.avg, top5.avg, batch_time.avg, data_time.avg


def save_checkpoint(state, is_best, filename='checkpoint.pth.tar', save_dir='.'):
    torch.save(state, os.path.join(save_dir, filename))
    if is_best:
        shutil.copyfile(os.path.join(save_dir, filename), os.path.join(save_dir, 'model_best.pth.tar'))


class AverageMeter(object):
    """Computes and stores the average and current value"""
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


def adjust_learning_rate(optimizer, epoch):
    """Sets the learning rate to the initial LR decayed by 10 every 30 epochs"""
    lr = args.lr * (0.1 ** (epoch // 30))
    for param_group in optimizer.param_groups:
        param_group['lr'] = lr

def adjust_learning_rate_multiopt(optimizer1, optimizer2, epoch, lr_decay_epoch=30):
    """Sets the learning rate to the initial LR decayed by 10 every 30 epochs"""
    lr = args.lr * (0.1 ** (epoch // lr_decay_epoch))
    for param_group in optimizer1.param_groups:
        param_group['lr'] = lr
    for param_group in optimizer2.param_groups:
        param_group['lr'] = lr
    return lr


def accuracy(output, target, topk=(1,)):
    """Computes the precision@k for the specified values of k"""
    with torch.no_grad():
        maxk = max(topk)
        batch_size = target.size(0)

        _, pred = output.topk(maxk, 1, True, True)
        pred = pred.t()
        correct = pred.eq(target.view(1, -1).expand_as(pred))

        res = []
        for k in topk:
            correct_k = correct[:k].view(-1).float().sum(0, keepdim=True)
            res.append(correct_k.mul_(100.0 / batch_size))
        return res


if __name__ == '__main__':
    main()
