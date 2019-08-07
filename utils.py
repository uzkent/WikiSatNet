import os
import re
import torch
import torchvision.transforms as transforms
import torchvision.datasets as torchdata
import torchvision.models as torchmodels
import numpy as np
from data_loader import WikiSatNet, fMoW
import pdb
import shutil

def save_args(__file__, args):
    shutil.copy(os.path.basename(__file__), args.cv_dir)
    with open(args.cv_dir+'/args.txt','w') as f:
        f.write(str(args))

def get_transforms():
    mean = [0.485, 0.456, 0.406]
    std = [0.229, 0.224, 0.225]
    transform_train = transforms.Compose([
       transforms.Resize(224),
       transforms.RandomCrop(224),
       transforms.ToTensor(),
       transforms.Normalize(mean, std)
    ])
    transform_test = transforms.Compose([
       transforms.Resize(224),
       transforms.CenterCrop(224),
       transforms.ToTensor(),
       transforms.Normalize(mean, std)
    ])

    return transform_train, transform_test

# Pick from the datasets available and the hundreds of models we have lying around depending on the requirements.
def get_dataset(train_csv, val_csv, pretrain=False):
    transform_train, transform_test = get_transforms()
    if pretrain:
        trainset = WikiSatNet(train_csv, transform_train)
        testset = WikiSatNet(val_csv, transform_test)
    else:
        trainset = fMoW(train_csv, transform_train)
        testset = fMoW(val_csv, transform_test)

    return trainset, testset

def set_parameter_requires_grad(model, feature_extracting):
    # When loading the models, make sure to call this function to update the weights
    if feature_extracting:
        for param in model.parameters():
            param.requires_grad = False

def get_model():
    dnet = torchmodels.densenet161(pretrained=True)
    set_parameter_requires_grad(dnet, False)
    num_ftrs = dnet.classifier.in_features
    dnet.classifier = torch.nn.Linear(num_ftrs, 300)

    return dnet
