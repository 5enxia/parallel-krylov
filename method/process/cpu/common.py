import numpy as np
from numpy.linalg import norm
from mpi4py import MPI

from krylov.method.common import _start, _end


def init(A, b, T=np.float64):
    x = np.zeros(b.size, T)
    b_norm = norm(b)
    N = b.size
    max_iter = N  # * 2
    residual = np.zeros(max_iter+1, T)
    num_of_solution_updates = np.zeros(max_iter+1, np.int)
    num_of_solution_updates[0] = 0
    return x, b_norm, N, max_iter, residual, num_of_solution_updates


def start(method_name='', k=None):
    _start(method_name, k)
    return MPI.Wtime()


def end(
    start_time, isConverged, num_of_iter, residual, residual_index,
    final_k=None
):
    elapsed_time = MPI.Wtime() - start_time
    _end(
        elapsed_time, isConverged, num_of_iter, residual, residual_index,
        final_k
    )
    return elapsed_time


def matvec(A, local_A, x, Ax, local_Ax, comm):
    """[summary]

    Args:
        A (np.ndarray): [行列(N * N)]
        local_A ([type]): [ローカル行列(local_N * N)]
        x (np.ndarray): [ベクトル]
        Ax (np.ndarray): [A.dot(x)]
        local_Ax (np.ndarray): [local_A.dot(x)]
        comm (): [MPI.COMM_WORLD()]

    Returns:
        [np.ndarray]: [演算結果]
    """
    comm.Bcast(x, root=0)
    comm.Scatter(A, local_A, root=0)
    local_Ax = np.dot(local_A, x)
    comm.Gather(local_Ax, Ax, root=0)
    return Ax


def vecvec(a, local_a, b, local_b, ab, local_ab, comm):
    """[summary]

    Args:
        a (np.ndarray): [ベクトル1]
        local_a ([type]): [ローカルベクトル1]
        b ([type]): [ベクトル2]
        local_b ([type]): [ローカルベクトル2]
        ab ([type]): [a.dot(b)]
        local_ab ([type]): [local_a.dot(local_b)]
        comm ([type]): [MPI.COMM_WORLD()]

    Returns:
        [type]: [description]
    """
    comm.Scatter(a, local_a, root=0)
    comm.Scatter(b, local_b, root=0)
    local_ab = np.dot(local_a, local_b)
    comm.Reduce(local_ab, ab, root=0)
    return ab
