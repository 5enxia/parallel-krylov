import sys
sys.path.append('../')
import numpy as np
import loader

# const
epsilon = 1e-10

# init
T = np.float64
directory = '../'
version = 'EFG'
N = 1081
ext = '.txt'
A = loader.matrixLoader(f'{directory}matrix_{version}-{N}{ext}', version, N, T)
b = loader.vectorLoader(f'{directory}vector_{version}-{N}{ext}',  version, N, T)
x = np.zeros(A.shape[0],T)
max_iter = b.shape


# residual
r = b - np.dot(A, x)
p = r.copy()


for k in max_iter:
    alpha = np.dot(r.T, p) / np.dot(p.T, np.dot(A, p))
    x = x + alpha * p
    r = r - alpha * np.dot(A, p)

    if np.linalg.norm(r) < epsilon:
        break

    beta = np.dot(r.T, r) / np.dot(r.T, r)

    p = r + beta * p