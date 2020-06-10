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
            nn.Linear(ndim, 64),
            nn.LeakyReLU(0.2),
            nn.Linear(64, 128),
            nn.LeakyReLU(0.2),
            nn.Linear(128, 64),
            nn.LeakyReLU(0.2),
            nn.Linear(64, 1)
        )

    def forward(self, X):
        validity = self.model(X)
        return validity
    
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

def to_onehot(target):
    # change the target to one-hot version
    Y = np.ravel(target.numpy()).astype(int)
    Y_train = np.zeros((Y.shape[0], Y.max()-Y.min()+1))
    Y_train[np.arange(Y.size), Y-Y.min()] = 1
    target_onehot =torch.from_numpy(Y_train.astype(np.float32))
    return target_onehot

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
    
def train(args, epoch, R_net, D_net, trainLoader, optimizer_R, optimizer_D, trainF, lr, device):
    R_net.train()
    D_net.train()
    MSEloss = nn.MSELoss()
    for batch_idx, (data, target) in enumerate(trainLoader):
        ones_label = Variable(torch.ones(data.shape[0], 1).to(device))
        zeros_label = Variable(torch.zeros(data.shape[0], 1).to(device))
        z = torch.randn(data.shape[0], args.latent_dim)
        z = Variable(torch.div(z,torch.t(torch.norm(z,p='fro',dim=1).repeat(args.latent_dim, 1))).to(device))
        data = Variable(data.to(device))

        # update Discriminator
        optimizer_D.zero_grad()
        w, _ = R_net(data)
        new_w = Variable(w.clone())
        D_real = torch.sigmoid(D_net(new_w))
        D_fake = torch.sigmoid(D_net(z))
        D_loss_real = F.binary_cross_entropy(D_real, ones_label)
        D_loss_fake = F.binary_cross_entropy(D_fake, zeros_label)
        D_loss = (D_loss_real + D_loss_fake)/2.
        optimizer_D.zero_grad()
        D_loss.backward()
        optimizer_D.step()
        
        # push forward particles
        w.detach_()
        w_t = Variable(w.clone(), requires_grad=True)
        d = -D_net(w_t)
        d.backward(torch.ones(w_t.shape[0],1).to(device),retain_graph=True)
        w = w + lr * w_t.grad
        
        # update Reducer
        optimizer_R.zero_grad()
        target_onehot = Variable(to_onehot(target).to(device))
        target = Variable(target.to(device))
        latent, output = R_net(data)
        Mloss = MSEloss(w, latent) # fit particles with Reducer
        dCor_loss = cor(latent, target_onehot, data.shape[0], device) # distance correlation
        R_loss = Mloss - dCor_loss
        R_loss.backward()
        optimizer_R.step()
        
        # calculate losses, error and original GAN loss
        D_nerual = torch.sigmoid(D_net(latent))
        OG_loss = F.binary_cross_entropy(D_nerual, zeros_label) # check the original GAN loss
        pred = output.data.max(1)[1]
        incorrect = pred.ne(target.data).cpu().sum()
        err = torch.tensor(100.)*incorrect/len(data)
    trainF.write('{},{},{}\n'.format(epoch, dCor_loss, err)) # log distance correlation
    trainF.flush()
    C_loss = F.nll_loss(output, target)
    print('Train Epoch: {}, Loss: {:.4f}, Error: {:.4f}, dCor_loss: {:.4f}, VG: {:.4f}, D: {:.2f}, OG: {:.2f}'.format(
        epoch, C_loss, err, dCor_loss, Mloss, D_loss, OG_loss))

def test(args, epoch, R_net, testLoader, optimizer, testF, device):
    R_net.eval()
    test_loss = 0
    incorrect = 0
    dCor_loss = 0
    with torch.no_grad():
        for data, target in testLoader:
            data = Variable(data.to(device))
            target_onehot = Variable(to_onehot(target).to(device))
            target = Variable(target.to(device))
            latent, output = R_net(data)
            dCor_loss += cor(latent, target_onehot, data.shape[0], device)
            test_loss += F.nll_loss(output, target).data
            pred = output.data.max(1)[1] 
            incorrect += pred.ne(target.data).cpu().sum()
    test_loss /= len(testLoader) 
    dCor_loss /= len(testLoader)
    nTotal = len(testLoader.dataset)
    err = torch.tensor(100.)*incorrect/nTotal
    print('\nTest set: Average loss: {:.4f}, dCor_loss: {:.4f}, Error: {}/{} ({:.0f}%)\n'.format(
        test_loss, dCor_loss, incorrect, nTotal, err))
    testF.write('{},{},{}\n'.format(epoch, dCor_loss, err))
    testF.flush()

def adjust_opt(optimizer, epoch):
    if epoch < 150: lr = 1e-1
    elif epoch == 150: lr = 1e-2
    elif epoch == 225: lr = 1e-3
    else: return
    
    for param_group in optimizer.param_groups:
        param_group['lr'] = lr

def scatter_plots(X, y):
    df = pd.DataFrame(data=X,columns=['Feature-one', 'Feature-two'])
    df['y'] = y
    sns.set_style("whitegrid")
    olm = sns.lmplot(
        x="Feature-one", y="Feature-two",
        hue="y",
        data=df,
        legend="full",
        fit_reg=False,
        scatter_kws={'alpha':0.3},
    )
    olm.set(xlim=(-1.5,1.51), ylim=(-1.5,1.51))