# basic functions
import os
import sys
import math
import numpy as np
import shutil
import setproctitle
import argparse
import pandas as pd
from numpy import genfromtxt
import matplotlib.pyplot as plt
from sklearn.preprocessing import StandardScaler
from sklearn.neighbors import KNeighborsClassifier
from sklearn.metrics import classification_report, confusion_matrix

# torch functions
import torch
import torch.optim as optim
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
parser.add_argument('--cuda_device', type=int, default=0, help='choose cuda device')
parser.add_argument('--no-cuda', action='store_true', help='if TRUE, cuda will not be used')
parser.add_argument('--path', help='path to load models')
parser.add_argument('--seed', type=int, default=1, help='random seed')
args = parser.parse_args()

args.cuda = not args.no_cuda and torch.cuda.is_available()
device = torch.device("cuda" if args.cuda else "cpu")
args.path = args.path or 'Results/MNIST'
setproctitle.setproctitle(args.path)

torch.manual_seed(args.seed)
if args.cuda:
    torch.cuda.manual_seed(args.seed)
    torch.cuda.set_device(args.cuda_device)

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

# nets setting
R_net = DenseNet(growthRate=12, depth=20, reduction=0.5,
                 bottleneck=True, ndim = args.latent_dim, nClasses=10)
D_net = Discriminator(ndim = args.latent_dim)

print('  + Number of params (R net) : {}'.format(
    sum([p.data.nelement() for p in R_net.parameters()])))
print('  + Number of params (D net) : {}'.format(
    sum([p.data.nelement() for p in D_net.parameters()])))

if args.cuda:
    R_net = R_net.cuda()
    D_net = D_net.cuda()

R_net.load_state_dict(torch.load(os.path.join(args.path, 'R.pt')))
D_net.load_state_dict(torch.load(os.path.join(args.path, 'D.pt')))
print('  + Models loaded')

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
print('  + Accuracy: %f' % acc)

# generate distance correlation plots
DDR = genfromtxt(os.path.join(args.path, 'test.csv'), delimiter=',')[:,1]
df = pd.DataFrame({'x': range(1, 301), 'DDR':DDR})
plt.style.use('seaborn-darkgrid')
my_dpi=200
plt.figure(figsize=(820/my_dpi, 680/my_dpi), dpi=my_dpi)
plt.plot(df['x'], df['DDR'], marker='', color='red', linewidth=1, alpha=0.4, label='DDR')
plt.legend(loc=4, ncol=2)
plt.xlim(0,310)
plt.title("", loc='left', fontsize=12, fontweight=0, color='orange')
plt.xlabel("Epoch")
plt.ylabel("Distance Correlation")
plt.savefig(os.path.join(args.path,'dcor_plot.png'))
print('done')