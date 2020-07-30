import torch
import torch.nn as nn
import torch.optim as optim
import torch.nn.functional as F
from torch.autograd import Variable
import numpy as np
import seaborn as sns
import pandas as pd

class Discriminator(nn.Module):
    def __init__(self, ndim):
        super(Discriminator, self).__init__()

        self.model = nn.Sequential(
            nn.Linear(ndim, 16),
            nn.LeakyReLU(0.2),
            nn.Linear(16, 8),
            nn.LeakyReLU(0.2),
            nn.Linear(8, 1)
        )

    def forward(self, X):
        validity = self.model(X)
        return validity
    
class Generator(nn.Module):
    def __init__(self, xdim, ndim):
        super(Generator, self).__init__()

        self.model = nn.Sequential(
            nn.Linear(xdim, 32),
            nn.LeakyReLU(0.2),
            nn.Linear(32, 16),
            nn.LeakyReLU(0.2),
            nn.Linear(16, 8),
            nn.LeakyReLU(0.2),
            nn.Linear(8, ndim),
        )
        
        self.fc = nn.Linear(ndim, 1)

    def forward(self, X):
        latent = self.model(X)
        out = self.fc(latent)
        return latent, out
    
def pairwise_distances(x, y=None):
    x_norm = (x**2).sum(1).view(-1, 1)
    if y is not None:
        y_norm = (y**2).sum(1).view(1, -1)
    else:
        y = x
        y_norm = x_norm.view(1, -1)

    dist = x_norm + y_norm - 2.0 * torch.mm(x, torch.transpose(y, 0, 1))
    return dist

def cor(X, Y, n, device):
    # to calculate distance correlation
    DX = pairwise_distances(X)
    DY = pairwise_distances(Y)
    J = (torch.eye(n) - torch.ones(n,n) / n).to(device)
    RX = J @ DX @ J
    RY = J @ DY @ J
    covXY = torch.mul(RX, RY).sum()/(n*n)
    covX = torch.mul(RX, RX).sum()/(n*n)
    covY = torch.mul(RY, RY).sum()/(n*n)
    return covXY/torch.sqrt(covX*covY)

def torch_tile(tensor, dim, n):
    """Tile n times along the dim axis"""
    if dim == 0:
        return tensor.unsqueeze(0).transpose(0,1).repeat(1,n,1).view(-1,tensor.shape[1])
    else:
        return tensor.unsqueeze(0).transpose(0,1).repeat(1,1,n).view(tensor.shape[0], -1)
    
def MedianDist(x):
    px = pairwise_distances(x).view(1,-1).squeeze()
    mdis = torch.sqrt(torch.median(px[px.nonzero().squeeze()]))
    return mdis

def Totensor(x, device):
    return torch.Tensor(x).to(device)

def Gram(X, sg, N, Q, device):
    sg2 = 2*sg*sg
    aa = torch.sum(X * X, 1).reshape(N,1)
    ab = torch.matmul(X, X.T)
    D = torch_tile(aa, 1, N)
    xx = torch.max(D + D.T - 2*ab, torch.zeros((N,N)).to(device))
    Gx = torch.exp(-xx/sg2)
    Kx = torch.matmul(Q, torch.matmul(Gx,Q))
    Kx = (Kx+Kx.T)/2
    return Kx

def HSIC_in(X, Z, Y, N, device, EPS=0.000001):
    X = (X-torch.mean(X))/torch.var(X)**(0.5)
    Z = (Z-torch.mean(Z))/torch.var(Z)**(0.5)
    sgx = MedianDist(X) 
    sgz = 0.5*sgx
    sgy = MedianDist(Y)
    I = torch.eye(N).to(device)
    Q = I - torch.ones(N).to(device)/N
    Kx = Gram(torch.cat((X,Z),1), sgx, N, Q, device)
    Ky = Gram(torch.cat((Y,Z),1), sgy, N, Q, device)
    Kz = Gram(Z, sgx, N, Q, device)
    Rx = torch.matmul(Kx, torch.inverse(Kx + EPS*N*I))
    Ry = torch.matmul(Ky, torch.inverse(Ky + EPS*N*I))
    Rz = torch.matmul(Kz, torch.inverse(Kz + EPS*N*I))
    term1 = torch.matmul(Ry, Rx)
    term2 = -2.*torch.matmul(term1, Rz)
    term3 = torch.matmul(torch.matmul(torch.matmul(Ry, Rz), Rx), Rz)
    HSIC = torch.trace(term1+term2+term3)
    return HSIC

def npLoader(Loader, net, device):
    # obtain the features and corresponding targets after dimension reduction
    X, y = next(iter(Loader))
    mb_size = X.shape[0]
    X = net(X.to(device))[0].cpu().detach().numpy()
    y = y.numpy()
    torch.cuda.empty_cache()
    for step, (X_t, y_t) in enumerate(Loader):
        X_t = net(X_t.cuda())[0].cpu().detach().numpy()
        y_t = y_t.numpy()
        X = np.concatenate((X, X_t))
        y = np.concatenate((y, y_t))
        torch.cuda.empty_cache()
    return X[mb_size:], y[mb_size:]

def train(args, epoch, net, D_net, trainLoader, optimizer, optimizer_D, zlr, device):
    net.train()
    D_net.train()
    nProcessed = 0
    MSEloss = nn.MSELoss()
    nTrain = len(trainLoader.dataset)
    for batch_idx, (data, target) in enumerate(trainLoader):
        ones_label = Variable(torch.ones(data.shape[0], 1).to(device))
        zeros_label = Variable(torch.zeros(data.shape[0], 1).to(device))
        z = Variable(torch.rand(data.shape[0], args.latent_dim).to(device))
        data = Variable(data.to(device))

        # Discriminator forward-loss-backward-update
        w, _ = net(data)
        new_w = Variable(w.clone())
        D_real = torch.sigmoid(D_net(new_w))
        D_fake = torch.sigmoid(D_net(z))
        D_loss_real = torch.nn.functional.binary_cross_entropy(D_real, ones_label)
        D_loss_fake = torch.nn.functional.binary_cross_entropy(D_fake, zeros_label)
        
        D_loss = (D_loss_real + D_loss_fake)/2.
        optimizer_D.zero_grad()
        D_loss.backward()
        optimizer_D.step()
        w.detach_()
        # Housekeeping - reset gradient
        w_t = Variable(w.clone(), requires_grad=True)
        d = -D_net(w_t)
        d.backward(torch.ones(w_t.shape[0],1).to(device),retain_graph=True)
        w = w + zlr * w_t.grad
        optimizer.zero_grad()
        latent, output = net(data)
        D_nerual = torch.sigmoid(D_net(latent))
        OG_loss = torch.nn.functional.binary_cross_entropy(D_nerual, zeros_label)
        Mloss = MSEloss(w, latent)
        d_loss = cor(latent, target.reshape(data.shape[0],1).to(device), data.shape[0], device)
        loss = Mloss - d_loss
#         loss = Mloss
        loss.backward()
        optimizer.step()

def test(args, epoch, R_net, testLoader, optimizer, device):
    R_net.eval()
    dCor_loss = 0
    with torch.no_grad():
        for data, target in testLoader:
            data = Variable(data.to(device))
            target = Variable(target.to(device))
            latent, output = R_net(data)
            dCor_loss += cor(latent, target, data.shape[0], device)
    dCor_loss /= len(testLoader)
    print('\nEpoch {}: Test set: dCor_loss: {:.4f}'.format(
         epoch, dCor_loss))
