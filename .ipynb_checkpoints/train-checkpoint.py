# basic functions
import os
import sys
import math
import numpy as np
import shutil
import setproctitle
import argparse
from sklearn.preprocessing import StandardScaler
from sklearn.neighbors import KNeighborsClassifier
from sklearn.metrics import classification_report, confusion_matrix

# torch functions
import torch
import torch.nn as nn
import torch.optim as optim
import torch.nn.functional as F
from torch.autograd import Variable
import torchvision.datasets as dset
import torchvision.transforms as transforms
from torchvision.utils import save_image
from torch.utils.data import DataLoader

# local functions
from densenet_mnist import DenseNet
from model import *

#------------------------------------------------------------------------------

# arguments setting
parser = argparse.ArgumentParser()
parser.add_argument('--batchSz', type=int, default=64, help='mini batch size')
parser.add_argument('--latent_dim', type=int, default=16, help='the dimension of latent space')
parser.add_argument('--nEpochs', type=int, default=300, help='the number of outter loop')
parser.add_argument('--cuda_device', type=int, default=0, help='choose cuda device')
parser.add_argument('--no-cuda', action='store_true', help='if TRUE, cuda will not be used')
parser.add_argument('--save', help='path to save results')
parser.add_argument('--seed', type=int, default=1, help='random seed')
args = parser.parse_args()

args.cuda = not args.no_cuda and torch.cuda.is_available()
device = torch.device("cuda" if args.cuda else "cpu")
args.save = args.save or 'Results/MNIST'
setproctitle.setproctitle(args.save)
torch.manual_seed(args.seed)
if args.cuda:
    torch.cuda.manual_seed(args.seed)
    torch.cuda.set_device(args.cuda_device)

if os.path.exists(args.save):
    shutil.rmtree(args.save)
os.makedirs(args.save, exist_ok=True)

# get dataloaders
trainTransform = transforms.Compose([
    transforms.ToTensor(),
    transforms.Normalize((0.1307,), (0.3081,))
])
testTransform = transforms.Compose([
    transforms.ToTensor(),
    transforms.Normalize((0.1307,), (0.3081,))
])

kwargs = {'num_workers': 1, 'pin_memory': True} if args.cuda else {}
train_set = dset.MNIST(root='mnist', train=True, download=True, 
                       transform=trainTransform)
test_set = dset.MNIST(root='mnist', train=False, download=True,
                 transform=testTransform)
trainLoader = DataLoader(train_set, batch_size=args.batchSz, 
                         shuffle=True, **kwargs)
testLoader = DataLoader(test_set,batch_size=args.batchSz, 
                        shuffle=False, **kwargs)

# nets and optimizers setting
R_net = DenseNet(growthRate=12, depth=20, reduction=0.5,
                 bottleneck=True, ndim = args.latent_dim, nClasses=10)
D_net = Discriminator(ndim = args.latent_dim)

print('  + Number of params (net) : {}'.format(
    sum([p.data.nelement() for p in R_net.parameters()])))
print('  + Number of params (Dnet) : {}'.format(
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
    lr = 0.1
    train(args, epoch, R_net, D_net, trainLoader, optimizer_R, optimizer_D, trainF, lr, device)
    test(args, epoch, R_net, testLoader, optimizer_R, testF, device)
    torch.save(R_net.state_dict(), os.path.join(args.save, 'R.pt'))
    torch.save(D_net.state_dict(), os.path.join(args.save, 'D.pt'))
trainF.close()
testF.close()

#------------------------------------------------------------------------------

# evaluate models
R_net.eval()
torch.cuda.empty_cache()
X_train, y_train = npLoader(trainLoader, R_net, device)
X_test, y_test = npLoader(testLoader, R_net, device)

scaler = StandardScaler()
scaler.fit(X_train)
X_train = scaler.transform(X_train)
X_test = scaler.transform(X_test)

# KNN for classification
classifier = KNeighborsClassifier(n_neighbors=5)
classifier.fit(X_train, y_train)
y_pred = classifier.predict(X_test)
print(confusion_matrix(y_test, y_pred))
print(classification_report(y_test, y_pred))

acc = 100 * np.sum(y_pred == y_test) / y_pred.shape
print('Accuracy: %f' % acc)
print('Done!')