# basic functions
import os
import sys
import math
import numpy as np
import argparse
import shutil
from math import sqrt
from sklearn.preprocessing import StandardScaler
from sklearn.model_selection import train_test_split
from sklearn.linear_model import LinearRegression
from sklearn.model_selection import KFold
from sklearn.metrics import mean_squared_error

# torch functions
import torch
import torch.nn as nn
import torch.optim as optim
import torch.nn.functional as F
from torch.autograd import Variable
import torchvision.transforms as transforms
from torch.utils.data import DataLoader, TensorDataset

# local functions
from model_reg import *

parser = argparse.ArgumentParser()
parser.add_argument('--batchSz', type=int, default=64, help='mini batch size')
parser.add_argument('--latent_dim', type=int, default=2, help='the dimension of latent space')
parser.add_argument('--nEpochs', type=int, default=300, help='the number of outter loop')
parser.add_argument('--cuda_device', type=int, default=0, help='choose cuda device')
parser.add_argument('--no-cuda', action='store_true', help='if TRUE, cuda will not be used')
parser.add_argument('--save', help='path to save results')
parser.add_argument('--seed', type=int, default=1, help='random seed')
parser.add_argument('--model', type=int, default=2, help='1: Model A; 2: Model B; 3: Model C')
parser.add_argument('--scenario', type=int, default=2, help='1: scenario 1 ; 2: scenario 2; 3: scenario 3')
args = parser.parse_args()

args.cuda = not args.no_cuda and torch.cuda.is_available()
device = torch.device("cuda" if args.cuda else "cpu")
args.save = args.save or 'work/sim_reg'

torch.manual_seed(args.seed)
if args.cuda:
    torch.cuda.manual_seed(args.seed)
    torch.cuda.set_device(args.cuda_device)

if os.path.exists(args.save):
    shutil.rmtree(args.save)
os.makedirs(args.save, exist_ok=True)

device = torch.device("cuda" if args.cuda else "cpu")
kwargs = {'num_workers': 1, 'pin_memory': True} if args.cuda else {}
n_sample = 5000

# Model type
if args.model == 1:
    X = np.random.randn(n_sample, 10)
elif args.model == 2:
    distributions = [
        {"type": np.random.normal, "kwargs": {"loc": 2, "scale": 1}},
        {"type": np.random.uniform, "kwargs": {"low": -1, "high": 1}},
        {"type": np.random.normal, "kwargs": {"loc": -2, "scale": 1}},
    ]
    coefficients = np.array([0.3, 0.3, 0.3])
    coefficients /= coefficients.sum() 
    num_distr = len(distributions)
    data = np.zeros((n_sample, num_distr,10))
    for idx, distr in enumerate(distributions):
        data[:, idx] = distr["type"](size=(n_sample,10), **distr["kwargs"])
    random_idx = np.random.choice(np.arange(num_distr), size=(n_sample,), p=coefficients)
    X = data[np.arange(n_sample), random_idx]
elif args.model == 3:
    mean = np.zeros(10)
    cov = 0.3*np.eye(10) + 0.7*np.multiply(np.ones((10,1)),np.ones((1,10)))
    X = np.random.multivariate_normal(mean, cov, n_sample)

# Scenario type
if args.scenario == 1:
    truth = (X[:,0]+X[:,1])**2+(1+np.exp(X[:,1]))**2
elif args.scenario == 2:
    truth = np.sin(np.pi*(X[:,0]+X[:,1])/10.)+ X[:,1]**2
elif args.scenario == 3:
    truth = (X[:,0]**2+X[:,1]**2)**0.5*np.log((X[:,0]**2+X[:,1]**2)**0.5)

eps = np.random.randn(n_sample, 1)
sigma = 0.25
y = truth.reshape(n_sample,1)+sigma*eps
X = X.astype(np.float32)
y = y.astype(np.float32)

indices = np.arange(X.shape[0])
kf = KFold(n_splits=5)
tup1=()
tup2=()
tup3=()

for train_index, test_index in kf.split(X):
    X_train, X_test = X[train_index], X[test_index]
    y_train, y_test = y[train_index], y[test_index]
    train_dat = TensorDataset(torch.from_numpy(X_train).float(), torch.from_numpy(y_train))
    trainLoader = DataLoader(train_dat, batch_size=args.batchSz, shuffle=True)
    test_dat = TensorDataset(torch.from_numpy(X_test).float(), torch.from_numpy(y_test))
    testLoader = DataLoader(test_dat, batch_size=args.batchSz, shuffle=False)
    testLoader_cor = DataLoader(test_dat, batch_size=len(test_dat), shuffle=False)
    D_net = Discriminator(ndim = args.latent_dim)
    net = Generator(xdim = X.shape[1], ndim = args.latent_dim)
    
    print('  + Number of params (net) : {}'.format(
        sum([p.data.nelement() for p in net.parameters()])))
    print('  + Number of params (Dnet) : {}'.format(
        sum([p.data.nelement() for p in D_net.parameters()])))
    if args.cuda:
        net = net.cuda()
        D_net = D_net.cuda()
        
    optimizer = optim.Adam(net.parameters(), weight_decay=1e-4)
    optimizer_D = optim.Adam(D_net.parameters(), weight_decay=1e-4)

    for epoch in range(1, args.nEpochs + 1):
        if epoch < 150: zlr = 3.0
        elif epoch == 150: zlr = 2.0
        elif epoch == 225: zlr = 1.0
        train(args, epoch, net, D_net,trainLoader, optimizer, optimizer_D, zlr, device)
        test(args, epoch, net, testLoader, optimizer, device)
        torch.save(net.state_dict(), os.path.join(args.save, 'R.pt'))
        torch.save(D_net.state_dict(), os.path.join(args.save, 'D.pt'))
        
    net.eval()
    X_test_t, y_test_t = next(iter(testLoader_cor))
    X_test_t, y_test_t = X_test_t.to(device), y_test_t.to(device)
    lant_t, _ = net(X_test_t)
    tup1 += (cor(lant_t, y_test_t, X_test_t.shape[0], device).item(),)
    tup2 += (HSIC_in(X_test_t,lant_t, y_test_t, X_test_t.shape[0], device).item(),)
    print(cor(lant_t, y_test_t, X_test_t.shape[0], device).item())
    print(HSIC_in(X_test_t,lant_t, y_test_t, X_test_t.shape[0], device).item())
    torch.cuda.empty_cache()
    X_train, y_train = npLoader(trainLoader, net, device)
    X_test, y_test = npLoader(testLoader, net, device)
    
    scaler = StandardScaler()
    scaler.fit(X_train)
    X_train = scaler.transform(X_train)
    X_test = scaler.transform(X_test)
    reg = LinearRegression().fit(X_train, y_train)
    y_pred = reg.predict(X_test)
    rms = sqrt(mean_squared_error(y_test, y_pred))
    tup3 += (rms,)
print("done!")
print('\nDDR DC \tmean: {:.4f}\tsd: {:.4f}\n'.format(np.mean(tup1),np.var(tup1)**(0.5)))
print('DDR HSIC \tmean: {:.4f}\tsd: {:.4f}\n'.format(np.mean(tup2),np.var(tup2)**(0.5)))
print('DDR MSE \tmean: {:.4f}\tsd: {:.4f}\n'.format(np.mean(tup3),np.var(tup3)**(0.5)))