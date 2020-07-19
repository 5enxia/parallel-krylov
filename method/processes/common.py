import numpy as np
from numpy.linalg import norm
from mpi4py import MPI

from ..common import _start, _end


def start(method_name='', k=None):
    _start(method_name, k)
    return MPI.Wtime()


def end(start_time, isConverged, num_of_iter, final_residual, final_k=None):
    elapsed_time = MPI.Wtime() - start_time
    _end(elapsed_time, isConverged, num_of_iter, final_residual, final_k)
    return elapsed_time


def init_mpi():
    comm = MPI.COMM_WORLD
    return comm, comm.Get_rank(), comm.Get_size()


def init_gpu(rank):
    import cupy as cp
    num_of_gpu = cp.cuda.runtime.getDeviceCount()
    cp.cuda.Device(rank % num_of_gpu).use()
    pool = cp.cuda.MemoryPool(cp.cuda.malloc_managed)
    cp.cuda.set_allocator(pool.malloc)
    return num_of_gpu


def init(A, b, num_of_process, T, pu):
    old_N = b.size
    num_of_append = ((num_of_process - (old_N % num_of_process)) % num_of_process)
    N = old_N + num_of_append
    local_N = N // num_of_process

    if num_of_append:
        A = np.append(A, np.zeros((old_N, num_of_append)), axis=1)  # 右に0を追加
        A = np.append(A, np.zeros((num_of_append, N)), axis=0)  # 下に0を追加
        b = np.append(b, np.zeros(num_of_append))  # 0を追加
    x = np.zeros(N, T)
    b_norm = norm(b)

    max_iter = old_N * 2
    residual = np.zeros(max_iter+1, T)
    num_of_solution_updates = np.zeros(max_iter+1, np.int)
    num_of_solution_updates[0] = 0

    if pu == 'gpu':
        import cupy as cp
        return cp.asarray(A), cp.asarray(b), cp.asarray(x), b_norm, N, max_iter, residual, num_of_solution_updates

    return A, b, x, b_norm, N, local_N, max_iter, residual, num_of_solution_updates
