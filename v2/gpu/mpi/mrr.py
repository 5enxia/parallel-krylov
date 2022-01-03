import cupy as cp
from cupy import dot
from cupy.linalg import norm
from mpi4py import MPI

from .common import start, finish, init, MultiGpu, init_mpi, calc_alloc_gpu


def mrr(A, b, epsilon, T):
    # MPI
    # rank 0-7
    # num_of_process = 8
    comm, rank, num_of_process = init_mpi()

    # GPU初期化
    begin, end = calc_alloc_gpu(rank, num_of_process)
    MultiGpu.init_gpu(begin, end, num_of_process)
    MultiGpu.joint_mpi(comm)

    # 初期化
    local_A, b, x, b_norm, N, max_iter, residual, num_of_solution_updates = init(A, b, T, rank, num_of_process, 16)

    MultiGpu.alloc(local_A, b, T)
    Ax = cp.zeros(N, T)
    Ar = cp.zeros(N, T)

    # 初期残差
    Ax = MultiGpu.dot(local_A, x, out=Ax)
    r = b - Ax
    residual[0] = norm(r) / b_norm

    # 初期反復
    i = 0
    if rank == 0:
        start_time = start(method_name='MrR + gpu + mpi')
    Ar = MultiGpu.dot(local_A, r, out=Ar)
    zeta = dot(r, Ar) / dot(Ar, Ar)
    MultiGpu.sync()
    y = zeta * Ar
    z = -zeta * r
    r -= y
    x -= z
    num_of_solution_updates[1] = 1
    i += 1

    # 反復計算
    while i < max_iter:
        # 収束判定
        residual[i] = norm(r) / b_norm
        if residual[i] < epsilon:
            isConverged = True
            break

        # 解の更新
        Ar = MultiGpu.dot(local_A, r, out=Ar)
        mu = dot(y, y)
        nu = dot(y, Ar)
        MultiGpu.sync()
        gamma = nu / mu
        s = Ar - gamma * y
        rs = dot(r, s)
        ss = dot(s, s)
        MultiGpu.sync()
        zeta = rs / ss
        eta = -zeta * gamma
        y = eta * y + zeta * Ar
        z = eta * z - zeta * r
        r -= y
        x -= z
        i += 1
        num_of_solution_updates[i] = i
    else:
        isConverged = False

    if rank == 0:
        elapsed_time = finish(start_time, isConverged, i, residual[i])
        return elapsed_time, num_of_solution_updates[:i+1], residual[:i+1]
    else:
        exit(0)
