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

from src.utils import *

import sys
srcFolder = os.path.join(os.path.abspath(os.path.dirname(__file__)), 'src')
sys.path.append(srcFolder)
import regressor
import gem

n_classes = 1000
arch = 'resnet50'
pretrained_model = 'data/model_best.pth.tar'
model = models.__dict__[arch]()
model,classifier = decomposeModel(model, n_classes, keep_pre_pooling=True)
model = torch.nn.DataParallel(model).cuda()

optimizer = torch.optim.SGD(model.parameters(), 0.1)
optimizer_reg = optim.SGD(classifier.parameters(), 0.1)
reg_net = regressor.Net(classifier, optimizer_reg, 
                        ref_size=1, 
                        backendtype=arch, 
                        dcl_offset=0, 
                        dcl_window=1, 
                        QP_margin=0.5)

checkpoint = torch.load(pretrained_model)
best_prec1 = checkpoint['best_prec1']
model.load_state_dict(checkpoint['state_dict'])
optimizer.load_state_dict(checkpoint['optimizer'])
# --------
optimizer_reg.load_state_dict(checkpoint['optimizer_reg'])
reg_net.load_state_dict(checkpoint['reg_net'])
reg_net.opt = optimizer_reg
reg_net.ref_size = checkpoint['reg_net_buffer_size']
reg_net.backendtype = checkpoint['reg_net_backendtype']
reg_net.dcl_window = checkpoint['reg_net_dcl_window']
reg_net.ref_cnt = checkpoint['reg_net_dcl_buffer_cnt']
reg_net.ref_data = checkpoint['reg_net_dcl_accum_grad'] if checkpoint['reg_net_dcl_accum_grad'] is not None else None
reg_net.stat_w1 = checkpoint['stat_ref_weight']

print('load the model learned at epoch {}'.format(checkpoint['epoch']))
print('validation accuracy is {:.2f}%'.format(checkpoint['best_prec1']))