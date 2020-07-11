import sys
import js
from scipy.linalg import toeplitz
import numpy as np
import cupy as cp
from cupy import dot
from cupy.linalg import norm
from mpi4py import MPI

# GPU Memory Settings
pool = cp.cuda.MemoryPool(cp.cuda.malloc_managed)
cp.cuda.set_allocator(pool.malloc)

# Numerical Conditions
## Accuracy
T = cp.float64
with open('condition.json') as f:
    params = json.load(f)
f.close()
epsilon, N, k, diag, sub_diag = params['epsilon'], params['N'], params['k'],params['diag'], -1
maxiter = N #* 2
## A, b, x, y
elements = np.zeros(N, T)
elements[0], elements[1] = diag, sub_diag
A_cpu, x_cpu = toeplitz(elements), np.zeros(N, T)
A_gpu, x_gpu = cp.asarray(A_cpu), cp.zeros(b_cpu)
b_cpu = cp.empty(N, T)
b_norm = norm(b_gpu)

# MPI Settings
comm = MPI.COMM_WORLD
rank = comm.Get_rank()
num_of_process = comm.Get_size()
num_of_local_row = N // num_of_process
## local A, local y, local l 
local_A_cpu, local_y_cpu = np.empty((num_of_local_row, N), T), np.empty((num_of_local_row, N), T)
local_A_gpu, local_y_gpu = cp.asarray(local_A_cpu), cp.asarray(local_y_cpu)
l_gpu = cp.empty(N, T)

# Metadata 
residuals = cp.zeros(maxiter + 1, T)
solution_updates = cp.zeros(maxiter + 1, cp.int)
k_history = cp.zeros(maxiter + 1, cp.int)

# Init Params
dif = 0
## Krylov Subspace
Ar = cp.empty((k + 3, N), T)
Ay = cp.empty((k + 2, N), T)
## scalar
alpha = cp.empty((2 * k + 3, 1), T)
beta = cp.empty((2 * k + 2, 1), T)
delta = cp.empty((2 * k + 1,1), T)

# Start 
if rank == 0:
    start_time = start()
    

# First Iter
Ar[0] = b - matvec(
residuals[0] = norm(Ar[0]) / b_norm
pre = presiduals[0]

# End
if rank == 0:
    end(start_time)