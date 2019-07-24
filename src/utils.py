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

def decomposeModel(model, n_classes, keep_pre_pooling=False):
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
        if not keep_pre_pooling:
            pooling = nn.MaxPool2d(kernel_size=tempList[-2].kernel_size,
                stride=tempList[-2].stride, 
                padding=tempList[-2].padding, 
                ceil_mode=tempList[-2].ceil_mode)
        else:
            pooling = tempList[-2]
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

def tensorOrFloat2Float(input):
    tmp = input if type(input).__name__ == 'float' else input.item()
    return tmp