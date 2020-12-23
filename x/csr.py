import time

import numpy as np
#import cupy as cp
#import cupyx as cpx
from scipy import sparse

N = 100

# Data
A = np.load('data/Meshless-Matrix-Reduced/matrix_EFG-10601.npy')
b = np.load('data/Meshless-Matrix-Reduced/matrix_EFG-10601.npy')

#A = cp.array(A_cpu)
#b = cp.array(b_cpu)
#compressed_A = cpx.scipy.sparse.csr_matrix(A)
#compressed_b = cpx.scipy.sparse.csr_matrix(b)
compressed_A = sparse.csr_matrix(A)
compressed_b = sparse.csr_matrix(b)

# Not Compress
start = time.perf_counter()
for i in range(N):
    A.dot(b)
print(time.perf_counter() - start)

# Compress
start = time.perf_counter()
for i in range(N):
    compressed_A.dot(b)
print(time.perf_counter() - start)
