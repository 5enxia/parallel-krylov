import numpy as np
import scipy
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


def krylov_base_start():
    return MPI.Wtime()


def krylov_base_finish(start_time):
    return MPI.Wtime() - start_time


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
    """[summary]

    Args:
        A(
            numpy.ndarray,
            cupy.ndarray,
            scipy.sparse.csr.csr_matrix,
            cupyx.scipy.sparse.csr.csr_matrix
        ): [係数行列]
        b(
            numpy.ndarray,
            cupy.ndarray,
        ): [右辺ベクトル]
        num_of_process(int): [mpiプロセス数]
        T ([type]): [精度]
    """

    old_N = b.size
    num_of_append = num_of_process - (old_N % num_of_process) # 足りない行を計算
    num_of_append = 0 if num_of_append == num_of_process else num_of_append
    N = old_N + num_of_append
    local_N = N // num_of_process

    x = None
    b_norm = None

    if pu == 'cpu':
        if isinstance(A, np.ndarray):
            if num_of_append:
                A = np.append(A, np.zeros((old_N, num_of_append)), axis=1)  # 右に0を追加
                A = np.append(A, np.zeros((num_of_append, N)), axis=0)  # 下に0を追加
        elif isinstance(A, scipy.sparse.csr.csr_matrix):
            from scipy.sparse import hstack, vstack, csr_matrix
            if num_of_append:
                A = hstack([A, csr_matrix((old_N, num_of_append))], 'csr') # 右にemptyを追加
                A = vstack([A, csr_matrix((num_of_append, N))], 'csr') # 下にemptyを追加
        if num_of_append:
            b = np.append(b, np.zeros(num_of_append))  # 0を追加
        x = np.zeros(N, T)
        b_norm = np.linalg.norm(b)
    else:
        import cupy as cp
        import cupyx
        if isinstance(A, np.ndarray):
            if num_of_append:
                A = cp.array(A)
                A = cp.append(A, cp.zeros((old_N, num_of_append)), axis=1)  # 右に0を追加
                A = cp.append(A, cp.zeros((num_of_append, N)), axis=0)  # 下に0を追加
        elif isinstance(A, scipy.sparse.csr.csr_matrix):
            from cupyx.scipy.sparse import hstack, vstack, csr_matrix
            if num_of_append:
                A = csr_matrix(A)
                A = hstack([A, csr_matrix((old_N, num_of_append))], 'csr') # 右にemptyを追加
                A = vstack([A, csr_matrix((num_of_append, N))], 'csr') # 下にemptyを追加
        if num_of_append:
            b = cp.append(b, cp.zeros(num_of_append))  # 0を追加
        x = cp.zeros(N, T)
        b_norm = cp.linalg.norm(b)

    max_iter = old_N * 2
    residual = np.zeros(max_iter+1, T)
    num_of_solution_updates = np.zeros(max_iter+1, np.int)
    num_of_solution_updates[0] = 0

    return A, b, x,\
        b_norm, N, local_N, max_iter, residual, num_of_solution_updates
