import numpy as np
from numpy.linalg import norm
from mpi4py import MPI

from krylov.method._common import _start, _end

def init(A, b, T = np.float64):
    x = np.zeros(b.size,T)
    y = np.empty(b.size,T)
    b_norm = norm(b)
    N = b.size
    max_iter = N#* 2
    residual = np.zeros(max_iter + 1,T)
    solution_updates = np.zeros(max_iter + 1, np.int)
    solution_updates[0] = 0
    return x, y, b_norm, N, max_iter, residual, solution_updates

def start(method_name = '', k = None):
    _start(method_name, k)
    return MPI.Wtime()

def end(start_time, isConverged, num_of_iter, residual, residual_index, final_k = None):
    elapsed_time = MPI.Wtime() - start_time
    _end(elapsed_time, isConverged, num_of_iter, residual, residual_index, final_k)
    return elapsed_time

def matvec(A, b,comm,T=np.float64):
    N = A.shape[0]
    num_of_process = comm.Get_size()

    y = np.empty(N, T)
    num_of_local_matrix_row = N // num_of_process 
    local_A = np.empty((num_of_local_matrix_row, N), dtype=T)
    comm.Bcast(b, root=0)
    comm.Scatter(A, local_A, root=0)
    local_y = np.dot(local_A, b)
    comm.Gather(local_y, y, root=0)
    return y

def vecvec(a,b,comm,T=np.float64):
    N = a.shape[0]
    num_of_process = comm.Get_size()

    y = np.empty(1,T)
    local_a = np.empty(N // num_of_process, T)
    local_b = np.empty(N // num_of_process, T)
    comm.Scatter(a, local_a, root=0)
    comm.Scatter(b, local_b, root=0)
    local_y = np.dot(local_a,local_b)
    comm.Reduce(local_y, y,root=0)
    return y

def vecmat(a,B,comm,T=np.float64):
    N = B.shape[0]
    num_of_process = comm.Get_size()

    y = np.empty(N,T) 
    num_of_local_matrix_row = N // num_of_process 
    local_B = np.empty((num_of_local_matrix_row, N), dtype=T)
    comm.Bcast(a, root=0)
    comm.Scatter(B, local_B, root=0)
    local_y = np.dot(a, local_B)
    comm.Gather(local_y, y, root=0)
    return y