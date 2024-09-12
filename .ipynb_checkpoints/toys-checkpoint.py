import numpy as np
import os
import matplotlib.pyplot as plt
from model import *
from sklearn.datasets import make_circles, make_moons
from sklearn.model_selection import train_test_split
from mpl_toolkits.mplot3d import Axes3D 

def toy_2d(args, sample_size = 10000):
    if args.dataset==1:
        X, y = make_circles(n_samples=sample_size, random_state=args.seed, noise=0.1, factor=0.5)
    elif args.dataset==2:
        X, y = make_moons(n_samples=sample_size, random_state=args.seed, noise=0.1)
    PX = X
    Py = y
    X = np.matmul(np.random.rand(20*20, 2), X.T).T
    X = X.reshape(sample_size, 1, 20, 20)
    X = X.astype(np.float32)
    indices = np.arange(sample_size)
    X_train, X_test, y_train, y_test, idx1, idx2 = train_test_split(X, y, indices, test_size=0.1, random_state=1)
    plt.figure(figsize=(16,16))
    scatter_plots(PX[idx2,], Py[idx2])
    plt.savefig(os.path.join(args.save, 'original.png'))
    return X_train, X_test, y_train, y_test

def toy_3d(args, single_size = 5000):
    fig = plt.figure(num=None, figsize=(8, 6), dpi=80, facecolor='w', edgecolor='k')
    ax = fig.add_subplot(111, projection='3d')
    from scipy.stats import multivariate_normal
    mean1 = [0, 0 ,0]
    cov1 = [[1, 0, 0], [0, 10, 0], [0, 0, 2]]
    mean2 = [25, 5 ,6]
    cov2 = [[4, 0, 0], [0, 5, 0], [0, 0, 1]]
    mean3 = [5, -10 ,-6]
    cov3 = [[5, 0, 0], [0, 3, 0], [0, 0, 5]]
    mean4 = [10, -1 ,-3]
    cov4 = [[4, 0, 0], [0, 1, 0], [0, 0, 1]]
    mean5 = [25, 3 ,1]
    cov5 = [[5, 0, 0], [0, 2, 0], [0, 0, 3]]
    mean6 = [20, -3 ,-1]
    cov6 = [[2, 0, 0], [0, 4, 0], [0, 0, 3]]
    xs, ys, zs = np.random.multivariate_normal(mean1, cov1, single_size).T
    count = np.zeros(single_size)
    data = np.array((xs,ys,zs,count)).T
    ax.scatter(xs, ys, zs, 'r')
    for c, mean, cov in [('b', mean2, cov2), ('c', mean3, cov3), ('y', mean4, cov4), ('k', mean5, cov5), ('p', mean6, cov6)]:
        xs, ys, zs = np.random.multivariate_normal(mean, cov, single_size).T
        count = count + 1
        data = np.concatenate((data,np.array((xs,ys,zs,count)).T))
        ax.scatter(xs, ys, zs, c)
    plt.savefig(os.path.join(args.save, 'original.png'))
    np.random.shuffle(data)
    sample_size = single_size * 6
    X = data[:,:3]
    y = data[:,3]
    X = np.matmul(np.random.rand(20*20, 3), X.T).T
    X = X.reshape(sample_size, 1, 20, 20)
    X = X.astype(np.float32)
    y = y.astype(np.long)
    indices = np.arange(sample_size)
    X_train, X_test, y_train, y_test, idx1, idx2 = train_test_split(X, y, indices, test_size=0.1, random_state=1)
    return X_train, X_test, y_train, y_test