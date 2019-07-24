# Copyright 2017-present, Facebook, Inc.
# All rights reserved.
#
# This source code is licensed under the license found in the
# LICENSE file in the root directory of this source tree.

import torch
import torch.nn as nn
import torch.optim as optim
from torch.autograd import Variable
import torch.nn.functional as F

import numpy as np
import quadprog
import sys

from scipy.optimize import linprog

def getWeights(parameters):
    weights = np.array([])
    for param in parameters():
        weights = np.concatenate((weights,param.data.cpu().view(-1).double().numpy()),axis=0)
    return weights

def compute_offsets(task, nc_per_task, is_cifar):
    """
        Compute offsets for cifar to determine which
        outputs to select for a given task.
    """
    if is_cifar:
        offset1 = task * nc_per_task
        offset2 = (task + 1) * nc_per_task
    else:
        offset1 = 0
        offset2 = nc_per_task
    return offset1, offset2


def store_grad(pp, grads, grad_dims, tid):
    """
        This stores parameter gradients of past tasks.
        pp: parameters
        grads: gradients
        grad_dims: list with number of parameters per layers
        tid: task id
    """
    # store the gradients
    grads[:, tid].fill_(0.0)
    cnt = 0
    for param in pp():
        if param.grad is not None:
            beg = 0 if cnt == 0 else sum(grad_dims[:cnt])
            en = sum(grad_dims[:cnt + 1])
            grads[beg: en, tid].copy_(param.grad.data.view(-1))
        cnt += 1


def overwrite_grad(pp, newgrad, grad_dims):
    """
        This is used to overwrite the gradients with a new gradient
        vector, whenever violations occur.
        pp: parameters
        newgrad: corrected gradient
        grad_dims: list storing number of parameters at each layer
    """
    cnt = 0
    for param in pp():
        if param.grad is not None:
            beg = 0 if cnt == 0 else sum(grad_dims[:cnt])
            en = sum(grad_dims[:cnt + 1])
            this_grad = newgrad[beg: en].contiguous().view(
                param.grad.data.size())
            param.grad.data.copy_(this_grad)
        cnt += 1

def project2cone2(gradient, memories, margin=0.5):
    """
        Solves the GEM dual QP described in the paper given a proposed
        gradient "gradient", and a memory of task gradients "memories".
        Overwrites "gradient" with the final projected update.

        input:  gradient, p-vector
        input:  memories, (t * p)-vector
        output: x, p-vector
    """
    memories_np = memories.cpu().t().double().numpy()
    gradient_np = gradient.cpu().contiguous().view(-1).double().numpy()
    t = memories_np.shape[0]
    P = np.dot(memories_np, memories_np.transpose())
    P = 0.5 * (P + P.transpose())
    q = np.dot(memories_np, gradient_np) * -1
    G = np.eye(t)
    h = np.zeros(t) + margin
    v = quadprog.solve_qp(P, q, G, h)[0]
    x = np.dot(v, memories_np) + gradient_np
    gradient.copy_(torch.Tensor(x).view(-1, 1))

class Net(nn.Module):
    def __init__(self,
                 classifier,
                 optimizer,
                 losscriterion=nn.CrossEntropyLoss(),
                 backendtype=None,
                 memory_strength=0.5,
                 n_memories=1,
                 useCUDA=True,
                 lr=0.01,
                 momentum=0.9,
                 weight_decay=1e-5):
        super(Net, self).__init__()
        #nl, nh = args.n_layers, args.n_hiddens
        nl, nh = 0,100
        self.margin = memory_strength

        self.net = classifier
        self.backendtype = backendtype

        self.ce = losscriterion

        self.opt = optimizer

        self.n_memories = n_memories
        self.gpu = useCUDA

        self.memory_raw_data = None
        total_dim = 0
        for param in self.parameters():
            total_dim += param.data.numel()
        self.memory_data = torch.FloatTensor(total_dim, self.n_memories).fill_(0.0)
        self.memory_labs = torch.FloatTensor(self.n_memories)
        if self.gpu:
            self.memory_data = self.memory_data.cuda()
            self.memory_labs = self.memory_labs.cuda()

        # allocate temporary synaptic memory
        self.grad_dims = []
        for param in self.parameters():
            self.grad_dims.append(param.data.numel())
        self.grads = torch.Tensor(sum(self.grad_dims), self.n_memories+1)

        if self.gpu:
            self.grads = self.grads.cuda()

        self.mem_cnt = 0
        self.stat_w1 = None

    def forward(self, x):
        x = x.view(x.size(0), -1)
        x = self.net(x)
        
        return x

    def observe(self, x, y, preNet_opt, criterion):

        if self.mem_cnt > 0 and self.mem_cnt == self.n_memories:
             # check each instance
            for tt in range(self.memory_data.size(0)):
                oldinput = self.memory_data[tt].unsqueeze(0)
                oldgt = self.memory_labs[tt]
                past_task = 0
                tmpret = self.forward(Variable(oldinput))
                memlab = Variable(torch.LongTensor([oldgt]).cuda())
                self.zero_grad()
                ptloss = self.ce(tmpret, memlab)
                ptloss.backward()
                store_grad(self.parameters, self.grads, self.grad_dims, tt)
            

        # now compute the grad on the current minibatch
        forward_output = self.forward(x)
        loss = self.ce(forward_output, y)

        preNet_opt.zero_grad()
        self.zero_grad()
        loss.backward()
        val_loss = loss.item()
        
        cur_idx = self.memory_data.size(0)
        store_grad(self.parameters, self.grads, self.grad_dims, cur_idx)

        cos_sim = [0.0,0.0,0.0] # for statistical purposes only, [congruency before grad modification, congruency after grad modification, magnitude of gradients]
        stat_ref_diretion = None
        if self.stat_w1 is not None:
            stat_ref_diretion = torch.from_numpy(getWeights(self.parameters)).type_as(self.stat_w1) - self.stat_w1
            stat_ref_diretion = stat_ref_diretion.type_as(self.grads)
        if stat_ref_diretion is not None and torch.norm(stat_ref_diretion) > 0:
            cos_sim[0] += torch.dot(stat_ref_diretion,self.grads[:, cur_idx]) / (torch.norm(stat_ref_diretion)*torch.norm(self.grads[:, cur_idx]))
            cos_sim[2] += torch.norm(stat_ref_diretion)
        if self.mem_cnt > 0 and self.mem_cnt == self.n_memories:
            indx = torch.cuda.LongTensor(list(range(0,cur_idx))) if self.gpu \
                    else torch.LongTensor(list(range(0,cur_idx)))
            dotp = torch.mm(self.grads[:, cur_idx].unsqueeze(0),
                            self.grads.index_select(1, indx))

            if self.mem_cnt >= 1 and (dotp < 0).sum() != 0:
                project2cone2(self.grads[:, cur_idx].unsqueeze(1),
                              self.grads.index_select(1, indx),self.margin)
                
                overwrite_grad(self.parameters, self.grads[:, cur_idx],
                               self.grad_dims)

                if stat_ref_diretion is not None and torch.norm(stat_ref_diretion) > 0:
                    cos_sim[1] += torch.dot(stat_ref_diretion,self.grads[:, cur_idx]) / (torch.norm(stat_ref_diretion)*torch.norm(self.grads[:, cur_idx]))

        self.opt.step()
        
        preNet_opt.step()
        
        if self.stat_w1 is None:
            self.stat_w1 = torch.from_numpy(getWeights(self.parameters))
        
        return loss, forward_output, cos_sim

    def resnet_forward(self,x):
        x = x.view(x.size(0), -1)
        x = self.net(x)
        return x

    def resnext_forward(self,x):
        x = x.view(x.size(0), -1)
        x = self.net(x)
        return x

    def densenet_forward(self,x):
        x = x.view(x.size(0), -1)
        x = self.net(x)
        return x

    def setRawMem(self, x, y):
        self.mem_cnt =  min(y.size(0),self.n_memories)
        self.memory_raw_data = x[:self.mem_cnt]
        self.memory_labs = y[:self.mem_cnt]

    def genMem(self, model):
        if self.n_memories > 0 and self.mem_cnt == self.n_memories:
            X = Variable(self.memory_raw_data).cuda()
            Y = Variable(self.memory_labs, requires_grad=False).cuda()
            self.memory_data = model(X).data.clone()
