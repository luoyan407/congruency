import argparse
import sys, os
from matplotlib import cm
from mpl_toolkits.mplot3d import Axes3D

import matplotlib.pyplot as plt
from matplotlib import rc
import numpy as np

import torch
import torch.nn as nn
import torch.nn.parallel
import torch.backends.cudnn as cudnn
import torch.optim as optim
import torch.utils.data as data
import torchvision.transforms as transforms
import torchvision.datasets as datasets

import quadprog

# if you have installed Times New Roman on your machine, uncomment the following line
# rc('font', **{'family':'serif','serif':['Times New Roman'], 'size':12})
rc('text', usetex=True)

class AverageMeter(object):
    """Computes and stores the average and current value
       Imported from https://github.com/pytorch/examples/blob/master/imagenet/main.py#L247-L262
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

def getWeights(parameters):
    weights = np.array([])
    for param in parameters():
        weights = np.concatenate((weights,param.data.cpu().view(-1).double().numpy()),axis=0)
    return weights

def getGrads(parameters):
    gradients = np.array([])
    for param in parameters():
        gradients = np.concatenate((gradients,param.grad.cpu().view(-1).double().numpy()),axis=0)
    return gradients

def is_pos_def(x):
    return np.all(np.linalg.eigvals(x)>0)

def project2cone2(gradient, constraints, margin=0.5):
    """
        Solves the DCL dual QP problem

        input:  gradient, p-vector
        input:  memories, (ref_size * p)-vector
        output: x, p-vector
    """
    constraints_np = constraints.cpu().t().double().numpy()
    gradient_np = gradient.cpu().contiguous().view(-1).double().numpy()
    t = constraints_np.shape[0]
    G = np.dot(constraints_np, constraints_np.transpose())
    G = 0.5 * (G + G.transpose())
    a = np.dot(constraints_np, gradient_np) * -1
    C = np.eye(t)
    b = np.zeros(t) + margin
    v = quadprog.solve_qp(G, a, C, b)[0]
    x = np.dot(v, constraints_np) + gradient_np
    gradient.copy_(torch.Tensor(x).view(-1, 1))

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

class SimpleModel(nn.Module):
    """
    A model to find a local minimum by gradient descent
    """
    def __init__(self, x, y, eff_win=-1):
        super(SimpleModel, self).__init__()
        self.x = nn.Parameter(x)
        self.y = nn.Parameter(y)
        self.xy0 = np.array([x.item(),y.item()])

    def forward(self):
        _, _, z = cost_func(self.x, self.y)
        return z

    def process(self, optimizer):
        v_info = [0.0,0.0,0.0]
        z = self.forward()
        x,y = self.x.item(), self.y.item()
        optimizer.zero_grad()
        z.backward()
        optimizer.step()

        grads = torch.from_numpy(getGrads(self.parameters))
        weights = torch.from_numpy(getWeights(self.parameters))
        ref_direction = (weights - torch.from_numpy(self.xy0))
        if torch.norm(ref_direction) > 0:
            tmp = torch.dot(ref_direction, grads) / (torch.norm(ref_direction)*torch.norm(grads))
            v_info[1] = tmp.item()
            tmp = torch.norm(ref_direction)
            v_info[2] = tmp.item()

        return x, y, z.item(), v_info

class SimpleDCLModel(nn.Module):
    """
    A model to find a local minimum by gradient descent and DCL
    """
    def __init__(self, x, y, eff_win=1000000):
        super(SimpleDCLModel, self).__init__()
        self.x = nn.Parameter(x)
        self.y = nn.Parameter(y)
        self.xy0 = np.array([x.item(),y.item()])
        self.p_ref = np.array([x.item(),y.item()])
        self.iter_count = 0
        self.reset_count = 1
        self.eff_win = eff_win

    def forward(self):
        _, _, z = cost_func(self.x, self.y)
        return z

    def process(self, optimizer):
        if self.eff_win > 0 and self.iter_count % self.eff_win == 0:
            self.p_ref = np.array([self.x.item(),self.y.item()])
        v_info = [0.0,0.0,0.0]
        z = self.forward()
        x,y = self.x.item(), self.y.item()

        # compute the accumulated gradient by subtracting the reference from current weight
        optimizer.zero_grad()
        z.backward()
        grads = torch.from_numpy(getGrads(self.parameters))
        weights = torch.from_numpy(getWeights(self.parameters))
        stat_ref_direction = (weights - torch.from_numpy(self.xy0))
        ref_direction = (weights - torch.from_numpy(self.p_ref))

        if torch.norm(stat_ref_direction) > 0:
            tmp = torch.dot(stat_ref_direction, grads) / (torch.norm(stat_ref_direction)*torch.norm(grads))
            v_info[0] = tmp.item()
            tmp = torch.norm(stat_ref_direction)
            v_info[2] = tmp.item()

        # test if ref direction matrix is positive definite
        ref_direction_mxt = ref_direction.unsqueeze(1)
        constraints_np = ref_direction_mxt.cpu().t().double().numpy()
        G = np.dot(constraints_np, constraints_np.transpose())
        G = 0.5 * (G + G.transpose())
        isPosDef = is_pos_def(G)
        neg_inner = torch.dot(grads,ref_direction).item() < 0
        if neg_inner and isPosDef:
            # solve the dual DCL QP 
            project2cone2(grads.unsqueeze(1),ref_direction_mxt)
            # write the updated gradient back in the variable for back-propagation
            overwrite_grad(self.parameters, grads, [1,1])
            if torch.norm(stat_ref_direction) > 0:
                tmp = torch.dot(stat_ref_direction, grads) / (torch.norm(stat_ref_direction)*torch.norm(grads))
                v_info[1] = tmp.item()

        optimizer.step()
        self.iter_count += 1

        return x, y, z.item(), v_info

def cost_func(x=None, y=None):
    '''Cost function.

    Args:
        x: None if placeholder tensor is used as input. Specify x to use x as input tensor.
        y: None if placeholder tensor is used as input. Specify y to use y as input tensor.

    Returns:
        Tuple (x, y, z) where x and y are input tensors and z is output tensor.
    '''

    # two local minima near (0, 0)
    z = -1 * torch.sin(x * x) * torch.cos(3 * y * y) * torch.exp(-(x * y) * (x * y)) - torch.exp(-(x + y) * (x + y))

    return x, y, z

# pyplot settings
plt.ion()
fig = plt.figure(figsize=(3, 2), dpi=300)
ax = fig.add_subplot(111, projection='3d')
plt.subplots_adjust(left=0, bottom=0, right=1, top=0.938, wspace=0, hspace=0)
params = {'legend.fontsize': 5,
          'legend.handlelength': 3}
plt.rcParams.update(params)
plt.axis('off')

# starting location for variables
x_i = 0.75
y_i = 1.0
steps = 250
xy_range = [-1.5, 1.5]

# visualize cost function as a contour plot
x_val = y_val = np.arange(xy_range[0], xy_range[1], 0.005, dtype=np.float32)
x_val_mesh, y_val_mesh = np.meshgrid(x_val, y_val)
x_val_mesh_flat = x_val_mesh.reshape([-1, 1])
y_val_mesh_flat = y_val_mesh.reshape([-1, 1])

_, _, z_val_mesh_flat = cost_func(torch.from_numpy(x_val_mesh_flat), torch.from_numpy(y_val_mesh_flat))
z_val_mesh_flat = z_val_mesh_flat.data.numpy()

z_val_mesh = z_val_mesh_flat.reshape(x_val_mesh.shape)
levels = np.arange(np.min(z_val_mesh_flat), np.max(z_val_mesh_flat), 0.05)
ax.contour(x_val_mesh, y_val_mesh, z_val_mesh, levels, alpha=.7, linewidths=0.4)
plt.draw()

# 3d plot camera zoom, angle
xlm = ax.get_xlim3d()
ylm = ax.get_ylim3d()
zlm = ax.get_zlim3d()
ax.set_xlim3d(xlm[0] * 0.5, xlm[1] * 0.5)
ax.set_ylim3d(ylm[0] * 0.5, ylm[1] * 0.5)
ax.set_zlim3d(zlm[0] * 0.5, zlm[1] * 0.5)
azm = ax.azim
ele = ax.elev + 40
ax.view_init(elev=ele, azim=azm)

parser = argparse.ArgumentParser(description='Convergence Visualization for the DCL Work')
parser.add_argument('--opt', type=str, default='adam', 
                    help='Optimizer type: gd, rmsprop, adam')
args = parser.parse_args()

z_file = 'z_results.npz'
output_path = 'figures'
if args.opt == 'gd':
    output_path = 'figures_gd'
    ops_param = np.array([[optim.SGD, 0.1, 'b', 'GD (lr=0.1)', SimpleModel, 0.1, 0.5, 10000, '--'],
                        [optim.SGD, 0.2, 'b', 'GD (lr=0.2)', SimpleModel, 0.1, 0.5, 10000, '-'],
                        [optim.SGD, 0.1, 'r', 'GD DCL (lr=0.1)', SimpleDCLModel, 0.1, 0.5, 2, '--'],
                        [optim.SGD, 0.2, 'r', 'GD DCL (lr=0.2)', SimpleDCLModel, 0.1, 0.5, 2, '-']])
elif args.opt == 'rmsprop':
    output_path = 'figures_rmsp'
    ops_param = np.array([[optim.RMSprop, 0.02, 'b', 'RMSP (lr=0.02)', SimpleModel, 0.0, 0.0, 10000, '--'],
                        [optim.RMSprop, 0.04, 'b', 'RMSP (lr=0.04)', SimpleModel, 0.0, 0.0, 10000, '-'],
                        [optim.RMSprop, 0.02, 'r', 'RMSP DCL (lr=0.02)', SimpleDCLModel, 0.0, 0.0, 2, '--'],
                        [optim.RMSprop, 0.04, 'r', 'RMSP DCL (lr=0.04)', SimpleDCLModel, 0.0, 0.0, 2, '-']])
elif args.opt == 'adam':
    output_path = 'figures_adam'
    ops_param = np.array([[optim.Adam, 0.05, 'b', 'Adam (lr=0.05)', SimpleModel, 0.0, 0.0, 10000, '--'],
                        [optim.Adam, 0.1, 'b', 'Adam (lr=0.1)', SimpleModel, 0.0, 0.0, 10000, '-'],
                        [optim.Adam, 0.05, 'r', 'Adam DCL (lr=0.05)', SimpleDCLModel, 0.0, 0.0, 6, '--'],
                        [optim.Adam, 0.1, 'r', 'Adam DCL (lr=0.1)', SimpleDCLModel, 0.0, 0.0, 6, '-']])

method_names = []
for i in range(ops_param.shape[0]):
    method_names.append(ops_param[i,3])
method_names = np.array(method_names)

if not os.path.exists(output_path):
    os.makedirs(output_path)

# use last location to draw a line to the current location
last_x, last_y, last_z = [], [], []
plot_cache = [None for _ in range(ops_param.shape[0])]
plot_line_cache = [None for _ in range(ops_param.shape[0])]

min_x, min_y, min_z = sys.float_info.max,sys.float_info.max,sys.float_info.max
best_method = ''
dirt_meters = []
magt_meters = []
models = []
optimizers = []
save_results = np.full((ops_param.shape[0],steps), sys.float_info.max)
for i in range(ops_param.shape[0]):
    x = torch.tensor(x_i)
    y = torch.tensor(y_i)
    sm = ops_param[i,4](x, y, ops_param[i,7])
    models.append(sm)
    if ops_param[i,3].startswith('RMSP'):
        optimizers.append(ops_param[i,0](sm.parameters(), 
            lr=ops_param[i,1], momentum=ops_param[i,5], weight_decay=ops_param[i,6], eps=1e-01))
    elif ops_param[i,3].startswith('Adam'):
        optimizers.append(ops_param[i,0](sm.parameters(), 
            lr=ops_param[i,1], betas=(0.8,0.999), eps=1e-1))
    else:
        optimizers.append(ops_param[i,0](sm.parameters(), lr=ops_param[i,1]))
    dirt_meters.append(AverageMeter())
    magt_meters.append(AverageMeter())

for iter in range(steps):
    for i, optimizer in enumerate(optimizers):
        x_val, y_val, z_val, v_info = models[i].process(optimizer)
        save_results[i, iter] = z_val
        dirt_meters[i].update(v_info[1])
        magt_meters[i].update(v_info[2])
        if min_z > z_val:
            min_x, min_y, min_z = x_val, y_val, z_val
            best_method = ops_param[i,3]

        if plot_cache[i]:
            plot_cache[i].remove()
        plot_cache[i] = ax.scatter(x_val, y_val, z_val, s=3, depthshade=True, label=ops_param[i, 3], color=ops_param[i, 2])
        # draw a line from the previous value
        if iter == 0:
            last_z.append(z_val)
            last_x.append(x_i)
            last_y.append(y_i)
        plot_line_cache[i], = ax.plot([last_x[i], x_val], [last_y[i], y_val], [last_z[i], z_val], linewidth=0.5, linestyle=ops_param[i, -1], color=ops_param[i, 2], label=ops_param[i, 3])
        last_x[i] = x_val
        last_y[i] = y_val
        last_z[i] = z_val

    if iter % 50 == 0:
        np.savez(os.path.join(output_path, z_file), z=save_results, names=method_names)

    if iter == 0:
        legend = ops_param[:, 3]
        plt.legend(plot_line_cache, legend, 
                bbox_to_anchor=(0.5,1.085), frameon=False,
                loc=9, ncol=2)


    plt.savefig(os.path.join(output_path,str(iter) + '.svg'))
    print('iteration: {}, ({},{},{}), {}'.format(iter, min_x, min_y, min_z, best_method))

    plt.pause(0.0001)
np.savez(os.path.join(output_path, z_file), z=save_results, names=method_names)
print("done")
