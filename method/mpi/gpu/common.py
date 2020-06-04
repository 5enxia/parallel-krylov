from mpi4py import MPI

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

    y = cp.empty(N, T)
    num_of_local_matrix_row = N // num_of_process 
    local_A = cp.empty((num_of_local_matrix_row, N), dtype=T)
    comm.Bcast(b, root=0)
    comm.Scatter(A, local_A, root=0)
    local_y = cp.dot(local_A, b)
    comm.Gather(local_y, y, root=0)
    return y

def vecvec(a,b,comm,T=cp.float64):
    N = a.shape[0]
    num_of_process = comm.Get_size()

    y = cp.empty(1,T)
    local_a = cp.empty(N // num_of_process, T)
    local_b = cp.empty(N // num_of_process, T)
    comm.Scatter(a, local_a, root=0)
    comm.Scatter(b, local_b, root=0)
    local_y = cp.dot(local_a,local_b)
    comm.Reduce(local_y, y,root=0)
    return y

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