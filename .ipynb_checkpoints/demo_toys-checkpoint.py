# basic functions
import os
import sys
import math
import numpy as np
import shutil
import setproctitle
import argparse
import matplotlib.pyplot as plt

# torch functions
import torch
import torch.optim as optim
import torchvision.transforms as transforms
from torchvision.utils import save_image
from torch.utils.data import DataLoader, TensorDataset
import torchvision.transforms as transforms

# local functions
from model import *
from toys import toy_2d, toy_3d
from densenet_sim import DenseNet

#------------------------------------------------------------------------------

# arguments setting
parser = argparse.ArgumentParser()
parser.add_argument('--batchSz', type=int, default=64, help='mini batch size')
parser.add_argument('--nEpochs', type=int, default=500, help='the number of outter loop')
parser.add_argument('--latent_dim', type=int, default=2, help='the dimension of latent space')
parser.add_argument('--no-cuda', action='store_true', help='if TRUE, cuda will not be used')
parser.add_argument('--cuda_device', type=int, default=0, help='choose cuda device')
parser.add_argument('--save', help='path to save results')
parser.add_argument('--seed', type=int, default=1, help='random seed')
parser.add_argument('--dataset', type=int, default=3, help='1: circles data; 2: moons data; 3: 3d guassian data')
args = parser.parse_args()

args.cuda = not args.no_cuda and torch.cuda.is_available()
device = torch.device("cuda" if args.cuda else "cpu")
args.save = args.save or 'Results/toys'
setproctitle.setproctitle(args.save)
 
torch.manual_seed(args.seed)
if args.cuda:
    torch.cuda.manual_seed(args.seed)
    torch.cuda.set_device(args.cuda_device)

if os.path.exists(args.save):
    shutil.rmtree(args.save)
os.makedirs(args.save, exist_ok=True)

kwargs = {'num_workers': 1, 'pin_memory': True} if args.cuda else {}

# get dataloaders
if (args.dataset == 1) or (args.dataset == 2):
        X_train, X_test, y_train, y_test = toy_2d(args, sample_size = 10000)
elif args.dataset == 3:
        X_train, X_test, y_train, y_test = toy_3d(args, single_size=5000)
train_dat = TensorDataset(torch.from_numpy(X_train).float(), torch.from_numpy(y_train))
trainLoader = DataLoader(train_dat, batch_size=args.batchSz, shuffle=True)
test_dat = TensorDataset(torch.from_numpy(X_test).float(), torch.from_numpy(y_test))
testLoader = DataLoader(test_dat, batch_size=args.batchSz, shuffle=False)

# nets and optimizers setting
R_net = DenseNet(growthRate=12, depth=10, reduction=0.5,
                        bottleneck=True, ndim = args.latent_dim, nClasses=10)
D_net = Discriminator(ndim = args.latent_dim)
print('  + Number of params (R net) : {}'.format(
    sum([p.data.nelement() for p in R_net.parameters()])))
print('  + Number of params (D net) : {}'.format(
    sum([p.data.nelement() for p in D_net.parameters()])))
if args.cuda:
    R_net = R_net.cuda()
    D_net = D_net.cuda()

optimizer_R = optim.Adam(R_net.parameters(), weight_decay=1e-4)
optimizer_D = optim.Adam(D_net.parameters(), weight_decay=1e-4)

trainF = open(os.path.join(args.save, 'train.csv'), 'w')
testF = open(os.path.join(args.save, 'test.csv'), 'w')

#------------------------------------------------------------------------------

# train models
for epoch in range(1, args.nEpochs + 1):
    if epoch < 50: zlr = 2.0
    elif epoch == 50: zlr = 1.5
    elif epoch == 150: zlr = 1.0
    train(args, epoch, R_net, D_net, trainLoader, optimizer_R, optimizer_D, trainF, zlr, device)
    test(args, epoch, R_net, testLoader, optimizer_R, testF, device)
    torch.save(R_net.state_dict(), os.path.join(args.save, 'R.pt'))
    torch.save(D_net.state_dict(), os.path.join(args.save, 'D.pt'))
    if epoch % 5 ==0:
        X_train, y_train = npLoader(trainLoader, R_net, device)
        X_test, y_test = npLoader(testLoader, R_net, device)
        scatter_plots(X_test, y_test)
        plt.savefig(os.path.join(args.save, 'latent_{}.png'.format(epoch)))
    
trainF.close()
testF.close()
print("Done!")