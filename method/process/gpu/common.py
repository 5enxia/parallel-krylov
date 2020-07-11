from mpi4py import MPI

import numpy as np
import cupy as cp 
from cupy.linalg import norm

def init(A, b, T = cp.float64):
    x = cp.zeros(b.size,T)
    y = cp.empty(b.size,T)
    b_norm = norm(b)
    N = b.size
    max_iter = N#* 2
    residual = cp.zeros(max_iter + 1,T)
    solution_updates = cp.zeros(max_iter + 1, cp.int)
    solution_updates[0] = 0
    return x, y, b_norm, N, max_iter, residual, solution_updates

def matvec(A, b,comm,T=cp.float64):
    N = A.shape[0]
    num_of_process = comm.Get_size()
    rank = comm.Get_rank()

    y = np.empty(N, T)
    num_of_local_matrix_row = N // num_of_process 
    local_A = np.empty((num_of_local_matrix_row, N), dtype=T)
    comm.Bcast(b.get(), root=0)
    comm.Scatter(A.get(), local_A, root=0)
    
    local_y = None
    with cp.cuda.Device(0):
        local_y = cp.dot(cp.array(local_A), cp.array(b))
    comm.Gather(local_y.get(), y, root=0)
    return cp.asarray(y)

def vecvec(a,b,comm,T=cp.float64):
    N = a.shape[0]
    num_of_process = comm.Get_size()
    rank = comm.Get_rank()

    y = np.empty(1,T)
    local_a = np.empty(N // num_of_process, T)
    local_b = np.empty(N // num_of_process, T)
    comm.Scatter(a.get(), local_a, root=0)
    comm.Scatter(b.get(), local_b, root=0)

    local_y = None
    with cp.cuda.Device(0):
        local_y = cp.dot(cp.array(local_a), cp.array(local_b))
    comm.Reduce(local_y.get(), y,root=0)
    return cp.asarray(y)

def vecmat(a,B,comm,T=cp.float64):
    N = B.shape[0]
    num_of_process = comm.Get_size()

    y = cp.empty(N,T) 
    num_of_local_matrix_row = N // num_of_process 
    local_B = cp.empty((num_of_local_matrix_row, N), dtype=T)
    comm.Bcast(a, root=0)
    comm.Scatter(B, local_B, root=0)
    local_y = cp.dot(a, local_B)
    comm.Gather(local_y, y, root=0)
    return y
