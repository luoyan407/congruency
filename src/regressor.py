import torch
import torch.nn as nn
import torch.optim as optim
from torch.autograd import Variable
import torch.nn.functional as F

import numpy as np
import quadprog
import sys

from scipy.optimize import linprog

def is_pos_def(x):
    # determine if x is positive definite
    return np.all(np.linalg.eigvals(x)>0)

def getGrads(parameters):
    # obtain gradients of weights
    grads = np.array([])
    for param in parameters():
        grads = np.concatenate((grads,param.grad.data.cpu().view(-1).double().numpy()),axis=0)
    return grads

def getWeights(parameters):
    # obtain weights
    weights = np.array([])
    for param in parameters():
        weights = np.concatenate((weights,param.data.cpu().view(-1).double().numpy()),axis=0)
    return weights

def store_grad(pp, grads, grad_dims):
    """
        This stores parameter gradients of past tasks.
        pp: parameters
        grads: gradients
        grad_dims: list with number of parameters per layers
    """
    # store the gradients
    grads.fill_(0.0)
    cnt = 0
    for param in pp():
        if param.grad is not None:
            beg = 0 if cnt == 0 else sum(grad_dims[:cnt])
            en = sum(grad_dims[:cnt + 1])
            grads[beg: en, 0].copy_(param.grad.data.view(-1))
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

def project2cone2(gradient, accumulated, margin=0.5):
    """
        Solves the dual QP problem.

        input:  gradient, p-vector
        input:  accumulated, (ref_num * p)-vector
        output: x, p-vector
    """
    accumulated_np = accumulated.cpu().t().double().numpy()
    gradient_np = gradient.cpu().contiguous().view(-1).double().numpy()
    t = accumulated_np.shape[0]
    G = np.dot(accumulated_np, accumulated_np.transpose())
    G = 0.5 * (G + G.transpose())
    a = np.dot(accumulated_np, gradient_np) * -1
    C = np.eye(t)
    b = np.zeros(t) + margin
    v = quadprog.solve_qp(G, a, C, b)[0]
    x = np.dot(v, accumulated_np) + gradient_np
    gradient.copy_(torch.Tensor(x).view(-1, 1))

class Net(nn.Module):
    def __init__(self,
                 classifier,
                 optimizer,
                 losscriterion=nn.CrossEntropyLoss(),
                 backendtype=None,
                 QP_margin=0.5,
                 ref_size=1,
                 useCUDA=True,
                 lr=0.01,
                 momentum=0.9,
                 weight_decay=1e-5,
                 dcl_offset=0,
                 dcl_window=0):
        super(Net, self).__init__()
        self.margin = QP_margin

        self.net = classifier
        self.backendtype = backendtype

        self.ce = losscriterion

        self.opt = optimizer

        self.ref_size = ref_size
        self.gpu = useCUDA

        self.ref_data = None
        if self.ref_size > 0:
            total_dim = 0
            for param in self.parameters():
                total_dim += param.data.numel()
            self.ref_data = torch.FloatTensor(total_dim, self.ref_size).fill_(0.0)
            if self.gpu:
                self.ref_data = self.ref_data.cuda()

        self.stat_w1 = None # for statistical purposes only

        # allocate space for gradients
        self.grad_dims = []
        for param in self.net.parameters():
            self.grad_dims.append(param.data.numel())
        self.grads = torch.Tensor(sum(self.grad_dims), 1)

        if self.gpu:
            self.grads = self.grads.cuda()

        self.ref_cnt = 0
        self.dcl_offset = dcl_offset
        self.iterCount = 0
        self.dcl_window = dcl_window
        self.ref_used = False

    def forward(self, x):
        x = x.view(x.size(0), -1)
        x = self.net(x)
        return x

    def observe(self, x, y, preNet_opt, criterion):

        # first compute the grad on the current minibatch
        forward_output = self.forward(x)
        # compute the loss
        loss = criterion(forward_output, y)

        preNet_opt.zero_grad()
        self.zero_grad()
        loss.backward()
        val_loss = loss.item()

        cur_idx = self.grads.size(1) - 1
        store_grad(self.parameters, self.grads, self.grad_dims)
        cos_sim = [0.0,0.0,0.0] # for statistical purposes only, [congruency before grad modification, congruency after grad modification, magnitude of gradients]
        stat_ref_diretion = None
        if self.stat_w1 is not None:
            stat_ref_diretion = torch.from_numpy(getWeights(self.parameters)).type_as(self.stat_w1) - self.stat_w1
            stat_ref_diretion = stat_ref_diretion.type_as(self.grads)
        if stat_ref_diretion is not None and torch.norm(stat_ref_diretion) > 0:
            cos_sim[0] += torch.dot(stat_ref_diretion,self.grads[:, 0]) / (torch.norm(stat_ref_diretion)*torch.norm(self.grads[:, 0]))
            cos_sim[2] += torch.norm(stat_ref_diretion)
        if self.ref_size > 0 and self.ref_cnt == self.ref_size:
            self.ref_used = True
            
            ref_diretion = torch.from_numpy(getWeights(self.parameters)).unsqueeze(1).expand_as(self.ref_data).type_as(self.ref_data) - self.ref_data
            # first test if the reference direction matrix is positive definite
            ref_np = ref_diretion.cpu().t().double().numpy()
            G = np.dot(ref_np, ref_np.transpose())
            G = 0.5 * (G + G.transpose())
            isPosDef = is_pos_def(G)
            if isPosDef:
                # compute the optimized gradient according to DCL constraints
                project2cone2(self.grads[:, 0].unsqueeze(1),
                              ref_diretion, self.margin)
                # write the optimized gradient back in position for back-propagation
                overwrite_grad(self.parameters, self.grads[:, 0],
                               self.grad_dims)

                if stat_ref_diretion is not None and torch.norm(stat_ref_diretion) > 0:
                    cos_sim[1] += torch.dot(stat_ref_diretion,self.grads[:, 0]) / (torch.norm(stat_ref_diretion)*torch.norm(self.grads[:, 0]))

        self.opt.step()
        
        preNet_opt.step()

        if self.dcl_window > 0 and self.iterCount % self.dcl_window == self.dcl_offset:
            self.reset()
        
        if self.ref_size > 0 and self.ref_cnt < self.ref_size:# and self.iterCount >= self.dcl_offset:
            self.ref_data[:,self.ref_cnt] = torch.from_numpy(getWeights(self.parameters))
            self.ref_cnt += 1
        if self.stat_w1 is None:
            self.stat_w1 = torch.from_numpy(getWeights(self.parameters))

        self.iterCount += 1

        return loss, forward_output, cos_sim

    def reset(self):
        self.ref_cnt =  0

    def resnet_forward(self,x):
        x = x.view(x.size(0), -1)
        x = self.net(x)
        return x

    def resnext_forward(self,x):
        x = F.avg_pool2d(x, 8, 1)
        x = x.view(-1, 1024)
        x = self.net(x)
        return x

    def densenet_forward(self,x):
        x = x.view(x.size(0), -1)
        x = self.net(x)
        return x

    def wideresnet_forward(self,x):
        x = F.avg_pool2d(x, 8)
        x = x.view(x.size(0),-1)
        x = self.net(x)
        return x