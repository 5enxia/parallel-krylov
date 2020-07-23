import numpy as np

from .common import start, end, init, init_mpi


def mrr(A, b, epsilon, T, pu):
    comm, rank, num_of_process = init_mpi()
    if pu == 'cpu':
        from numpy.linalg import norm
        from .cpu import init_matvec, init_vecvec, mpi_matvec, mpi_vecvec1, mpi_vecvec2
    else:
        import cupy as cp
        from cupy.linalg import norm
        from .common import init_gpu
        init_gpu(rank)
        from .gpu import init_matvec, init_vecvec, mpi_matvec, mpi_vecvec1, mpi_vecvec2

    # 共通初期化
    A, b, x, b_norm, N, local_N, max_iter, residual, num_of_solution_updates = init(A, b, num_of_process, T, pu)
    local_A, Ax, local_Ax = init_matvec(N, local_N, T)
    local_a, local_b = init_vecvec(local_N, T)
    if pu == 'cpu':
        comm.Scatter(A, local_A)
    else:
        local_A_cpu = np.empty((local_N, N), T)
        comm.Scatter(A, local_A_cpu)
        local_A = cp.asarray(local_A_cpu)

    # 初期残差
    r = b - mpi_matvec(local_A, x, Ax, local_Ax, comm)
    residual[0] = norm(r) / b_norm

    # 初期反復
    if rank == 0:
        start_time = start(method_name='MrR')
    Ar = mpi_matvec(local_A, r, Ax, local_Ax, comm)
    zeta = mpi_vecvec2(r, Ar, local_a, local_b, comm) / mpi_vecvec1(Ar, local_a, comm)
    y = zeta * Ar
    z = -zeta * r
    r -= y
    x -= z
    num_of_solution_updates[1] = 1
    i = 1

    # 反復計算
    while i < max_iter:
        # 収束判定
        residual[i] = norm(r) / b_norm
        isConverged = np.array([residual[i] < epsilon], bool)
        comm.Bcast(isConverged)
        if isConverged:
            break

        # 解の更新
        Ar = mpi_matvec(local_A, r, Ax, local_Ax, comm)
        nu = mpi_vecvec2(y, Ar, local_a, local_b, comm)
        gamma = nu / mpi_vecvec1(y, local_a, comm)
        s = Ar - gamma * y
        zeta = mpi_vecvec2(r, s, local_a, local_b, comm) / mpi_vecvec1(s, local_a, comm)
        eta = -zeta * gamma
        y = eta * y + zeta * Ar
        z = eta * z - zeta * r
        r -= y
        x -= z
        i += 1
        num_of_solution_updates[i] = i
    else:
        isConverged = False
        residual[i] = norm(r) / b_norm

    if rank == 0:
        elapsed_time = end(start_time, isConverged, i, residual[i])
        return elapsed_time, num_of_solution_updates[:i+1], residual[:i+1]
    else:
        exit(0)
