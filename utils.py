import os
import re
import torch
import torchvision.transforms as transforms
import torchvision.datasets as torchdata
import torchvision.models as torchmodels
import numpy as np
from data_loader import CustomDatasetFromImages
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
def get_dataset(root='data/'):
    transform_train, transform_test = get_transforms()
    trainset = CustomDatasetFromImages(root+'/wikipedia/train.csv', transform_train)
    testset = CustomDatasetFromImages(root+'/wikipedia/val.csv', transform_test)

    return trainset, testset

def set_parameter_requires_grad(model, feature_extracting):
    # When loading the models, make sure to call this function to update the weights
    if feature_extracting:
        for param in model.parameters():
            param.requires_grad = False

def get_model():

    rnet = torchmodels.resnet34(pretrained=True)
    set_parameter_requires_grad(rnet, False)
    num_ftrs = rnet.fc.in_features
    rnet.fc = torch.nn.Linear(num_ftrs, 300)

    return rnet
