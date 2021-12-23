import cupy as cp
from cupy import dot
from cupy.linalg import norm
from mpi4py import MPI

from .common import start, finish, init, MultiGpu, init_mpi, calc_alloc_gpu


def cg(A, b, epsilon, T):
    # MPI
    # rank 0-7
    # num_of_process = 8
    comm, rank, num_of_process = init_mpi()

    # GPU初期化
    begin, end = calc_alloc_gpu(rank, num_of_process)
    MultiGpu.init_gpu(begin, end, num_of_process)

    # 初期化
    local_A, b, x, b_norm, N, max_iter, residual, num_of_solution_updates = init(A, b, T, rank, num_of_process, 16)

    MultiGpu.alloc(local_A, b, T)
    Ax = cp.empty(N, T)
    v = cp.empty(N, T)

    # 初期残差
    comm.Allgather(MultiGpu.dot(local_A, x), Ax)
    r = b - Ax
    p = r.copy()
    gamma = dot(r, r)

    # 反復計算
    i = 0

    if rank == 0:
        start_time = start(method_name='cg + gpu + mpi')

    while i < max_iter:
        # 収束判定
        residual[i] = norm(r) / b_norm
        if residual[i] < epsilon:
            isConverged = True
            break

        # 解の更新
        comm.Allgather(MultiGpu.dot(local_A, p), v)
        sigma = dot(p, v)
        alpha = gamma / sigma
        x += alpha * p
        r -= alpha * v
        old_gamma = gamma.copy()
        gamma = dot(r, r)
        beta = gamma / old_gamma
        p = r + beta * p
        i += 1
        num_of_solution_updates[i] = i
    else:
        isConverged = False

    if rank == 0:
        elapsed_time = finish(start_time, isConverged, i, residual[i])
    return elapsed_time, num_of_solution_updates[:i+1], residual[:i+1]
