import os
from tensorboard_logger import configure, log_value
import torch
import torch.autograd as autograd
from torch.autograd import Variable
import torch.utils.data as torchdata
import torch.nn as nn
import torch.nn.functional as F
import numpy as np
import tqdm
import utils
import torch.optim as optim
import pdb
from torch.distributions import Multinomial, Bernoulli
import torch.backends.cudnn as cudnn
cudnn.benchmark = True

import argparse
parser = argparse.ArgumentParser(description='Transfer Learning')
parser.add_argument('--lr', type=float, default=1e-4, help='learning rate')
parser.add_argument('--data_dir', default='data/', help='data directory')
parser.add_argument('--cv_dir', default='cv/tmp/', help='checkpoint directory (models and logs are saved here)')
parser.add_argument('--train_csv', default='cv/tmp/', help='train csv directory')
parser.add_argument('--val_csv', default='cv/tmp/', help='validation csv directory')
parser.add_argument('--batch_size', type=int, default=256, help='batch size')
parser.add_argument('--epoch_step', type=int, default=10000, help='epochs after which lr is decayed')
parser.add_argument('--max_epochs', type=int, default=10000, help='total epochs to run')
parser.add_argument('--load', help='Checkpoints for the pretrained model')
parser.add_argument('--parallel', action ='store_true', default=False, help='use multiple GPUs for training')

args = parser.parse_args()

if not os.path.exists(args.cv_dir):
    os.system('mkdir ' + args.cv_dir)
utils.save_args(__file__, args)

def train(epoch):

    net.train()
    matches, losses = [], []
    for batch_idx, (inputs, targets) in tqdm.tqdm(enumerate(trainloader), total=len(trainloader)):

        inputs, targets = Variable(inputs), Variable(targets).cuda(async=True)
        if not args.parallel:
    	    inputs = inputs.cuda()

        v_inputs = Variable(inputs.data, volatile=True)

        preds = net.forward(v_inputs)

        _, pred_idx = preds.max(1)
        match = (pred_idx==targets).data

        loss = F.cross_entropy(preds, targets)

        optimizer.zero_grad()
        loss.backward()
        optimizer.step()

        matches.append(match.cpu())
        losses.append(loss.cpu())

    accuracy = torch.cat(matches, 0).float().mean()
    loss = torch.stack(losses).mean()
    log_str = 'E: %d | A: %.3f | L: %.3f'%(epoch, accuracy, loss)
    print log_str

    log_value('train_accuracy', accuracy, epoch)
    log_value('train_loss', loss, epoch)

def test(epoch):

    net.eval()
    matches = []
    for batch_idx, (inputs, targets) in tqdm.tqdm(enumerate(testloader), total=len(testloader)):

        inputs, targets = Variable(inputs, volatile=True), Variable(targets).cuda(async=True)
        if not args.parallel:
            inputs = inputs.cuda()

        v_inputs = Variable(inputs.data, volatile=True)

        preds = net.forward(v_inputs)

        _, pred_idx = preds.max(1)
        match = (pred_idx==targets).data

        matches.append(match.cpu())

    accuracy = torch.cat(matches, 0).float().mean()
    log_str = 'TS: %d | A: %.3f'%(epoch, accuracy)
    print log_str

    log_value('train_accuracy', accuracy, epoch)

    # save the model parameters
    net_state_dict = net.module.state_dict() if args.parallel else net.state_dict()

    state = {
      'state_dict': net_state_dict,
      'epoch': epoch,
      'acc': accuracy
    }
    torch.save(state, args.cv_dir+'/ckpt_E_%d_A_%.3f'%(epoch, accuracy))

#--------------------------------------------------------------------------------------------------------#
trainset, testset = utils.get_dataset(args.train_csv, args.val_csv)
trainloader = torchdata.DataLoader(trainset, batch_size=args.batch_size, shuffle=True, num_workers=16)
testloader = torchdata.DataLoader(testset, batch_size=args.batch_size, shuffle=False, num_workers=4)
net = utils.get_model()
if args.parallel:
    net = nn.DataParallel(net)
net.cuda()

if args.load:
    checkpoint = torch.load(args.load)
    torch.load_state_dict(checkpoint)

start_epoch = 0
optimizer = optim.Adam(net.parameters(), lr=args.lr)
configure(args.cv_dir+'/log', flush_secs=5)
for epoch in range(start_epoch, start_epoch+args.max_epochs+1):
    train(epoch)
    if epoch % 1 == 0:
        test(epoch)
    lr_scheduler.step()
