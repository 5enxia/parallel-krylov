import numpy as np

from .common import start, end as finish, init, init_mpi


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
    rsss = cp.empty(2, T)
    numu = cp.empty(2, T)
    # cpu
    Ar_cpu = np.empty(N, T)
    y_cpu = np.empty(N, T)
    rsss_cpu = np.empty(2, T)
    numu_cpu = np.empty(2, T)

    # 初期残差
    comm.Allgather(A[begin:end].dot(x).get(), Ax)
    r = b - cp.asarray(Ax)
    residual[0] = norm(r) / b_norm

    # 初期反復
    if rank == 0:
        start_time = start(method_name='MrR')
    # local_Ar = A[begin:end].dot(r)
    # comm.Allgather(local_Ar.get(), Ar_cpu)
    Ar[begin:end] = A[begin:end].dot(r)
    comm.Allgather(Ar[begin:end].get(), Ar_cpu)
    Ar = cp.asarray(Ar_cpu)
    # rsss[0] = r[begin:end].dot(local_Ar)
    # rsss[1] = local_Ar.dot(local_Ar)
    rsss[0] = r[begin:end].dot(Ar[begin:end])
    rsss[1] = Ar[begin:end].dot(Ar[begin:end])
    comm.Allreduce(rsss.get(), rsss_cpu)
    rsss = cp.asarray(rsss_cpu)
    zeta = rsss[0] / rsss[1]
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
        # local_Ar = A[begin:end].dot(r)
        # comm.Allgather(local_Ar.get(), Ar_cpu)
        Ar[begin:end] = A[begin:end].dot(r)
        comm.Allgather(Ar[begin:end].get(), Ar_cpu)
        Ar = cp.asarray(Ar_cpu)
        comm.Scatter(y.get(), y_cpu[begin:end])
        y[begin:end] = cp.asarray(y_cpu[begin:end])
        # numu[0] = y[begin:end].dot(local_Ar)
        numu[0] = y[begin:end].dot(Ar[begin:end])
        numu[1] = y[begin:end].dot(y[begin:end])
        comm.Allreduce(numu.get(), numu_cpu)
        numu = cp.asarray(numu_cpu)
        gamma = numu[0] / numu[1]
        s = Ar - gamma * y
        rsss[0] = r[begin:end].dot(s[begin:end])
        rsss[1] = s[begin:end].dot(s[begin:end])
        comm.Allreduce(rsss.get(), rsss_cpu)
        rsss = cp.asarray(rsss_cpu)
        zeta = rsss[0] / rsss[1]
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
        pass
    else:
        if rank == 0:
            return _mrr_gpu(A, b, epsilon, T, pu)
        else:
            _mrr_gpu(A, b, epsilon, T, pu)
