import numpy as np

from .common import start, end as finish, init, init_mpi


def _mrr_cpu(A, b, epsilon, T, pu):
    from numpy.linalg import norm

    # 共通初期化
    comm, rank, num_of_process = init_mpi()
    A, b, x, b_norm, N, local_N, max_iter, residual, num_of_solution_updates = init(A, b, num_of_process, T, pu)
    begin, end = rank * local_N, (rank+1) * local_N
    comm.Scatter(A, A[begin:end])

    # 初期化
    Ax = np.empty(N, T)
    Ar = np.empty(N, T)
    s = np.empty(N, T)
    rs = np.empty(1, T)
    ss = np.empty(1, T)
    nu = np.empty(1, T)
    mu = np.empty(1, T)

    # 初期残差
    comm.Gather(A[begin:end].dot(x), Ax)
    r = b - Ax
    residual[0] = norm(r) / b_norm

    # 初期反復
    if rank == 0:
        start_time = start(method_name='MrR')
    comm.Bcast(r)
    local_Ar = A[begin:end].dot(r)
    comm.Gather(local_Ar, Ar)
    comm.Reduce(r[begin:end].dot(local_Ar), rs)
    comm.Reduce(local_Ar.dot(local_Ar), ss)
    zeta = rs / ss
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
        comm.Bcast(r)
        local_Ar = A[begin:end].dot(r)
        comm.Gather(local_Ar, Ar)
        comm.Scatter(y, y[begin:end])
        comm.Reduce(y[begin:end].dot(local_Ar), nu)
        comm.Reduce(y[begin:end].dot(y[begin:end]), mu)
        gamma = nu / mu
        s = Ar - gamma * y
        comm.Scatter(s, s[begin:end])
        comm.Reduce(r[begin:end].dot(s[begin:end]), rs)
        comm.Reduce(s[begin:end].dot(s[begin:end]), ss)
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
        residual[i] = norm(r) / b_norm

    if rank == 0:
        elapsed_time = finish(start_time, isConverged, i, residual[i])
        return elapsed_time, num_of_solution_updates[:i+1], residual[:i+1]
    else:
        exit(0)


def _mrr_gpu(A, b, epsilon, T, pu):
    comm, rank, num_of_process = init_mpi()
    import cupy as cp
    from cupy.linalg import norm
    from .common import init_gpu
    init_gpu(rank)
    from .gpu import init_matvec, init_vecvec, mpi_matvec, mpi_vecvec1, mpi_vecvec2

    # 共通初期化
    A, b, x, b_norm, N, local_N, max_iter, residual, num_of_solution_updates = init(A, b, num_of_process, T, pu)
    local_A, Ax, local_Ax = init_matvec(N, local_N, T)
    local_a, local_b = init_vecvec(local_N, T)
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
        elapsed_time = finish(start_time, isConverged, i, residual[i])
        return elapsed_time, num_of_solution_updates[:i+1], residual[:i+1]
    else:
        exit(0)


def mrr(A, b, epsilon, T, pu):
    comm, rank, num_of_process = init_mpi()
    if pu == 'cpu':
        if rank == 0:
            return _mrr_cpu(A, b, epsilon, T, pu)
        else:
            _mrr_cpu(A, b, epsilon, T, pu)
    else:
        if rank == 0:
            return _mrr_gpu(A, b, epsilon, T, pu)
        else:
            _mrr_gpu(A, b, epsilon, T, pu)
