import numpy as np
import scipy
from numpy.linalg import norm

from mpi4py import MPI

from ..common import _start, _finish


def start(method_name='', k=None):
    _start(method_name, k)
    return MPI.Wtime()


def finish(start_time, isConverged, num_of_iter, final_residual, final_k=None):
    elapsed_time = MPI.Wtime() - start_time
    _finish(elapsed_time, isConverged, num_of_iter, final_residual, final_k)
    return elapsed_time


def init_mpi():
    comm = MPI.COMM_WORLD
    return comm, comm.Get_rank(), comm.Get_size()


def init(A, b, T, rank, num_of_process):
    # 追加する要素数を算出
    old_N = b.size
    num_of_append = num_of_process - (old_N % num_of_process) # 足りない行を計算
    num_of_append = 0 if num_of_append == num_of_process else num_of_append
    N = old_N + num_of_append
    local_N = N // num_of_process
    begin, end = rank * local_N, (rank+1) * local_N

    ## A
    if num_of_append:
        # データをパディングする
        if isinstance(A, np.ndarray):
            if num_of_append:
                A = np.append(A, np.zeros((old_N, num_of_append)), axis=1)  # 右に0を追加
                A = np.append(A, np.zeros((num_of_append, N)), axis=0)  # 下に0を追加
        elif isinstance(A, scipy.sparse.csr.csr_matrix):
            from scipy.sparse import hstack, vstack, csr_matrix
            if num_of_append:
                A = hstack([A, csr_matrix((old_N, num_of_append))], 'csr') # 右にemptyを追加
                A = vstack([A, csr_matrix((num_of_append, N))], 'csr') # 下にemptyを追加
    local_A = A[begin:end]

    ## b
    if num_of_append:
        b = np.append(b, np.zeros(num_of_append))  # 0を追加
    b_norm = np.linalg.norm(b)

    # x
    x = np.zeros(N, T)

    # その他パラメータ
    max_iter = old_N * 2
    residual = np.zeros(max_iter+1, T)
    num_of_solution_updates = np.zeros(max_iter+1, np.int)
    num_of_solution_updates[0] = 0


    return local_A, b, x, b_norm, N, max_iter, residual, num_of_solution_updates