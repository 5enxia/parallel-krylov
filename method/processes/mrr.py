import numpy as np

from .common import start, end as finish, init, init_mpi


def _mrr_cpu(A, b, epsilon, T, pu):
    from numpy.linalg import norm

    # 共通初期化
    comm, rank, num_of_process = init_mpi()
    A, b, x, b_norm, N, local_N, max_iter, residual, num_of_solution_updates = init(A, b, num_of_process, T, pu)
    begin, end = rank * local_N, (rank+1) * local_N

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
    import cupy as cp
    from cupy.linalg import norm

    from .common import init_gpu

    # 共通初期化
    comm, rank, num_of_process = init_mpi()
    init_gpu(rank)
    A, b, x, b_norm, N, local_N, max_iter, residual, num_of_solution_updates = init(A, b, num_of_process, T, pu)
    begin, end = rank * local_N, (rank+1) * local_N

    # 初期化
    Ax = np.empty(N, T)
    Ar = cp.empty(N, T)
    s = cp.empty(N, T)
    rs = cp.empty(1, T)
    ss = cp.empty(1, T)
    nu = cp.empty(1, T)
    mu = cp.empty(1, T)
    # cpu
    Ar_cpu = np.empty(N, T)
    rs_cpu = np.empty(1, T)
    ss_cpu = np.empty(1, T)
    nu_cpu = np.empty(1, T)
    mu_cpu = np.empty(1, T)

    # 初期残差
    comm.Allgather(A[begin:end].dot(x).get(), Ax)
    r = b - cp.asarray(Ax)
    residual[0] = norm(r) / b_norm

    # 初期反復
    if rank == 0:
        start_time = start(method_name='MrR')
    Ar = A[begin:end].dot(r)
    comm.Allgather(Ar.get(), Ar_cpu)
    Ar = cp.asarray(Ar_cpu)
    comm.Allreduce(r[begin:end].dot(Ar[begin:end]).get(), rs_cpu)
    comm.Allreduce(Ar[begin:end].dot(Ar[begin:end]).get(), ss_cpu)
    rs = cp.asarray(rs_cpu)
    ss = cp.asarray(ss_cpu)
    zeta = rs / ss
    y = zeta * Ar
    z = -zeta * r
    r -= y
    x -= z
    i = 1
    num_of_solution_updates[1] = 1

    # 反復計算
    while i < max_iter:
        # 収束判定
        residual[i] = norm(r) / b_norm
        isConverged = residual[i] < epsilon
        if isConverged:
            break

        # 解の更新
        comm.Allgather(A[begin:end].dot(r).get(), Ar_cpu)
        Ar = cp.asarray(Ar_cpu)
        comm.Allreduce(y[begin:end].dot(Ar[begin:end]).get(), nu_cpu)
        comm.Allreduce(y[begin:end].dot(y[begin:end]).get(), mu_cpu)
        nu = cp.asarray(nu_cpu)
        mu = cp.asarray(mu_cpu)
        gamma = nu / mu
        s = Ar - gamma * y
        comm.Allreduce(r[begin:end].dot(s[begin:end]).get(), rs_cpu)
        comm.Allreduce(s[begin:end].dot(s[begin:end]).get(), ss_cpu)
        rs = cp.asarray(rs_cpu)
        ss = cp.asarray(ss_cpu)
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
