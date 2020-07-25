import numpy as np

from .common import start, end as finish, init, init_mpi


def _kskipmrr_cpu(A, b, epsilon, k, T, pu):
    from numpy.linalg import norm

    # 共通初期化
    comm, rank, num_of_process = init_mpi()
    A, b, x, b_norm, N, local_N, max_iter, residual, num_of_solution_updates = init(A, b, num_of_process, T, pu)
    begin, end = rank * local_N, (rank+1) * local_N

    # 初期化
    Ax = np.empty(N, T)
    # Ar = xp.zeros((k + 3, N), T)
    # Ay = xp.zeros((k + 2, N), T)
    Ar = np.zeros((k + 3 + 1, N), T)
    Ay = np.zeros((k + 2 + 1, N), T)
    rAr = np.zeros(1, T)
    ArAr = np.zeros(1, T)
    alpha = np.zeros(2*k + 3, T)
    beta = np.zeros(2*k + 2, T)
    delta = np.zeros(2*k + 1, T)

    # local
    # local_Ar = xp.zeros(local_N, T)
    # local_Ay = xp.zeros(local_N, T)
    local_Ar = np.zeros((2, N), T)
    local_Ay = np.zeros((2, N), T)
    local_alpha = np.zeros(2*k + 3, T)
    local_beta = np.zeros(2*k + 2, T)
    local_delta = np.zeros(2*k + 1, T)
    
    # 初期残差
    comm.Gather(A[begin:end].dot(x), Ax)
    Ar[0] = b - Ax
    residual[0] = norm(Ar[0]) / b_norm

    # 初期反復
    if rank == 0:
        start_time = start(method_name='k-skip MrR', k=k)
    comm.Bcast(Ar[0])
    Ar[1][begin:end] = A[begin:end].dot(Ar[0])
    comm.Gather(Ar[1][begin:end], Ar[1])
    comm.Reduce(Ar[0][begin:end].dot(Ar[1][begin:end]), rAr)
    comm.Reduce(Ar[1][begin:end].dot(Ar[1][begin:end]), ArAr)
    zeta = rAr / ArAr
    Ay[0] = zeta * Ar[1]
    z = -zeta * Ar[0]
    Ar[0] -= Ay[0]
    x -= z
    i = 1
    index = 1
    num_of_solution_updates[1] = 1

    # 反復計算
    while i < max_iter:
        # 収束判定
        residual[index] = norm(Ar[0]) / b_norm
        isConverged = np.array([residual[index] < epsilon], bool)
        comm.Bcast(isConverged)
        if isConverged:
            break

        # 事前計算
        # for j in range(1, k + 2):
        #     Ar[j] = mpi_matvec(local_A, Ar[j-1], Ax, local_Ax, comm)
        # for j in range(1, k + 1):
        #     Ay[j] = mpi_matvec(local_A, Ay[j-1], Ax, local_Ax, comm)
        for j in range(1, (k + 2) + 1, 2):
            comm.Bcast(Ar[j-1])
            local_Ar[0][begin:end] = A[begin:end].dot(Ar[j-1])
            local_Ar[1] = A[begin:end].T.dot(local_Ar[0][begin:end])
            comm.Reduce(local_Ar, Ar[j:j+2])
        for j in range(1, (k + 1) + 1, 2):
            comm.Bcast(Ay[j-1])
            local_Ay[0][begin:end] = A[begin:end].dot(Ay[j-1])
            local_Ay[1] = A[begin:end].T.dot(local_Ay[0][begin:end])
            comm.Reduce(local_Ay, Ay[j:j+2])
        comm.Bcast(Ar)
        comm.Bcast(Ay)
        for j in range(2*k + 3):
            jj = j//2
            local_alpha[j] = Ar[jj][begin:end].dot(Ar[jj + j % 2][begin:end])
        for j in range(1, 2*k + 2):
            jj = j//2
            local_beta[j] = Ay[jj][begin:end].dot(Ar[jj + j % 2][begin:end])
        for j in range(2*k + 1):
            jj = j//2
            local_delta[j] = Ay[jj][begin:end].dot(Ay[jj + j % 2][begin:end])
        comm.Reduce(local_alpha, alpha)
        comm.Reduce(local_beta, beta)
        comm.Reduce(local_delta, delta)

        # MrRでの1反復(解と残差の更新)
        d = alpha[2] * delta[0] - beta[1] ** 2
        zeta = alpha[1] * delta[0] / d
        eta = -alpha[1] * beta[1] / d
        Ay[0] = eta * Ay[0] + zeta * Ar[1]
        z = eta * z - zeta * Ar[0]
        Ar[0] -= Ay[0]
        comm.Bcast(Ar[0])
        comm.Gather(A[begin:end].dot(Ar[0]), Ar[1])
        x -= z

        # MrRでのk反復
        for j in range(k):
            delta[0] = zeta ** 2 * alpha[2] + eta * zeta * beta[1]
            alpha[0] -= zeta * alpha[1]
            delta[1] = eta ** 2 * delta[1] + 2 * eta * zeta * beta[2] + zeta ** 2 * alpha[3]
            beta[1] = eta * beta[1] + zeta * alpha[2] - delta[1]
            alpha[1] = -beta[1]
            for l in range(2, 2 * (k - j) + 1):
                delta[l] = eta ** 2 * delta[l] + 2 * eta * zeta * beta[l+1] + zeta ** 2 * alpha[l + 2]
                tau = eta * beta[l] + zeta * alpha[l + 1]
                beta[l] = tau - delta[l]
                alpha[l] -= tau + beta[l]
            # 解と残差の更新
            d = alpha[2] * delta[0] - beta[1] ** 2
            zeta = alpha[1] * delta[0] / d
            eta = -alpha[1] * beta[1] / d
            Ay[0] = eta * Ay[0] + zeta * Ar[1]
            z = eta * z - zeta * Ar[0]
            Ar[0] -= Ay[0]
            comm.Bcast(Ar[0])
            comm.Gather(A[begin:end].dot(Ar[0]), Ar[1])
            x -= z

        i += (k + 1)
        index += 1
        num_of_solution_updates[index] = i
    else:
        isConverged = False
        residual[index] = norm(Ar[0]) / b_norm

    if rank == 0:
        elapsed_time = finish(start_time, isConverged, i, residual[index])
        return elapsed_time, num_of_solution_updates[:index+1], residual[:index+1]
    else:
        exit(0)


def _kskipmrr_gpu(A, b, epsilon, k, T, pu):
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
    # Ar = xp.zeros((k + 3, N), T)
    # Ay = xp.zeros((k + 2, N), T)
    Ar = cp.zeros((k + 3 + 1, N), T)
    Ay = cp.zeros((k + 2 + 1, N), T)
    rAr = cp.empty(1, T)
    ArAr = cp.empty(1, T)
    alpha = cp.zeros(2*k + 3, T)
    beta = cp.zeros(2*k + 2, T)
    delta = cp.zeros(2*k + 1, T)
    # local
    # local_Ar = xp.zeros(local_N, T)
    # local_Ay = xp.zeros(local_N, T)
    local_Ar = cp.zeros((2, N), T)
    local_Ay = cp.zeros((2, N), T)
    # cpu
    Ar_cpu = np.zeros((k + 3 + 1, N), T)
    Ay_cpu = np.zeros((k + 2 + 1, N), T)
    rAr_cpu = np.empty(1, T)
    ArAr_cpu = np.empty(1, T)
    alpha_cpu = np.zeros(2*k + 3, T)
    beta_cpu = np.zeros(2*k + 2, T)
    delta_cpu = np.zeros(2*k + 1, T)
    
    # 初期残差
    comm.Gather(A[begin:end].dot(x).get(), Ax)
    Ar[0] = b - cp.asarray(Ax)
    residual[0] = norm(Ar[0]) / b_norm

    # 初期反復
    if rank == 0:
        start_time = start(method_name='k-skip MrR', k=k)
    Ar_cpu[0] = Ar[0].get()
    comm.Bcast(Ar_cpu[0])
    Ar[0] = cp.asarray(Ar_cpu[0])
    Ar[1][begin:end] = A[begin:end].dot(Ar[0])
    comm.Gather(Ar[1][begin:end].get(), Ar_cpu[1])
    Ar[1] = cp.asarray(Ar_cpu[1])
    comm.Reduce(Ar[0][begin:end].dot(Ar[1][begin:end]).get(), rAr_cpu)
    comm.Reduce(Ar[1][begin:end].dot(Ar[1][begin:end]).get(), ArAr_cpu)
    rAr = cp.asarray(rAr_cpu)
    ArAr = cp.asarray(ArAr_cpu)
    zeta = rAr / ArAr
    Ay[0] = zeta * Ar[1]
    z = -zeta * Ar[0]
    Ar[0] -= Ay[0]
    x -= z
    i = 1
    index = 1
    num_of_solution_updates[1] = 1

    # 反復計算
    while i < max_iter:
        # 収束判定
        residual[index] = norm(Ar[0]) / b_norm
        isConverged = np.array([residual[index] < epsilon], bool)
        comm.Bcast(isConverged)
        if isConverged:
            break

        # 事前計算
        # for j in range(1, k + 2):
        #     Ar[j] = mpi_matvec(local_A, Ar[j-1], Ax, local_Ax, comm)
        # for j in range(1, k + 1):
        #     Ay[j] = mpi_matvec(local_A, Ay[j-1], Ax, local_Ax, comm)
        Ar_cpu = Ar.get()
        Ay_cpu = Ay.get()
        # for j in range(1, (k + 2) + 1, 2):
        comm.Bcast(Ar_cpu[0])
        for j in range(1, k + 2):
            # comm.Bcast(Ar_cpu[j-1])
            Ar[j-1] = cp.asarray(Ar_cpu[j-1])
            # local_Ar[0][begin:end] = A[begin:end].dot(Ar[j-1])
            # local_Ar[1] = A[begin:end].T.dot(local_Ar[0][begin:end])
            # comm.Reduce(local_Ar.get(), Ar_cpu[j:j+2])
            # comm.Gather(A[begin:end].dot(Ar[j-1]).get(), Ar_cpu[j])
            comm.AllGather(A[begin:end].dot(Ar[j-1]).get(), Ar_cpu[j])
        # for j in range(1, (k + 1) + 1, 2):
        comm.Bcast(Ay_cpu[0])
        for j in range(1, k + 1):
            # comm.Bcast(Ay_cpu[j-1])
            Ay[j-1] = cp.asarray(Ay_cpu[j-1])
            # local_Ay[0][begin:end] = A[begin:end].dot(Ay[j-1])
            # local_Ay[1] = A[begin:end].T.dot(local_Ay[0][begin:end])
            # comm.Reduce(local_Ay.get(), Ay_cpu[j:j+2])
            # comm.Gather(A[begin:end].dot(Ay[j-1]).get(), Ay_cpu[j])
            comm.AllGather(A[begin:end].dot(Ay[j-1]).get(), Ay_cpu[j])
        # comm.Bcast(Ar_cpu)
        # comm.Bcast(Ay_cpu)
        Ar = cp.asarray(Ar_cpu)
        Ay = cp.asarray(Ay_cpu)
        for j in range(2*k + 3):
            jj = j//2
            alpha[j] = Ar[jj][begin:end].dot(Ar[jj + j % 2][begin:end])
        for j in range(1, 2*k + 2):
            jj = j//2
            beta[j] = Ay[jj][begin:end].dot(Ar[jj + j % 2][begin:end])
        for j in range(2*k + 1):
            jj = j//2
            delta[j] = Ay[jj][begin:end].dot(Ay[jj + j % 2][begin:end])
        comm.Reduce(alpha.get(), alpha_cpu)
        comm.Reduce(beta.get(), beta_cpu)
        comm.Reduce(delta.get(), delta_cpu)
        alpha = cp.asarray(alpha_cpu)
        beta = cp.asarray(beta_cpu)
        delta = cp.asarray(delta_cpu)

        # MrRでの1反復(解と残差の更新)
        d = alpha[2] * delta[0] - beta[1] ** 2
        zeta = alpha[1] * delta[0] / d
        eta = -alpha[1] * beta[1] / d
        Ay[0] = eta * Ay[0] + zeta * Ar[1]
        z = eta * z - zeta * Ar[0]
        Ar[0] -= Ay[0]
        Ar_cpu[0] = Ar[0].get()
        comm.Bcast(Ar_cpu[0])
        Ar[0] = cp.asarray(Ar_cpu[0])
        comm.Gather(A[begin:end].dot(Ar[0]).get(), Ar_cpu[1])
        Ar[1] = cp.asarray(Ar_cpu[1])
        x -= z

        # MrRでのk反復
        for j in range(k):
            delta[0] = zeta ** 2 * alpha[2] + eta * zeta * beta[1]
            alpha[0] -= zeta * alpha[1]
            delta[1] = eta ** 2 * delta[1] + 2 * eta * zeta * beta[2] + zeta ** 2 * alpha[3]
            beta[1] = eta * beta[1] + zeta * alpha[2] - delta[1]
            alpha[1] = -beta[1]
            for l in range(2, 2 * (k - j) + 1):
                delta[l] = eta ** 2 * delta[l] + 2 * eta * zeta * beta[l+1] + zeta ** 2 * alpha[l + 2]
                tau = eta * beta[l] + zeta * alpha[l + 1]
                beta[l] = tau - delta[l]
                alpha[l] -= tau + beta[l]
            # 解と残差の更新
            d = alpha[2] * delta[0] - beta[1] ** 2
            zeta = alpha[1] * delta[0] / d
            eta = -alpha[1] * beta[1] / d
            Ay[0] = eta * Ay[0] + zeta * Ar[1]
            z = eta * z - zeta * Ar[0]
            Ar[0] -= Ay[0]
            Ar_cpu[0] = Ar[0].get()
            comm.Bcast(Ar_cpu[0])
            Ar[0] = cp.asarray(Ar_cpu[0])
            comm.Gather(A[begin:end].dot(Ar[0]).get(), Ar_cpu[1])
            Ar[1] = cp.asarray(Ar_cpu[1])
            x -= z

        i += (k + 1)
        index += 1
        num_of_solution_updates[index] = i
    else:
        isConverged = False
        residual[index] = norm(Ar[0]) / b_norm

    if rank == 0:
        elapsed_time = finish(start_time, isConverged, i, residual[index])
        return elapsed_time, num_of_solution_updates[:index+1], residual[:index+1]
    else:
        exit(0)


def kskipmrr(A, b, epsilon, k, T, pu):
    comm, rank, num_of_process = init_mpi()
    if pu == 'cpu':
        if rank == 0:
            return _kskipmrr_cpu(A, b, epsilon, k, T, pu)
        else:
            _kskipmrr_cpu(A, b, epsilon, k, T, pu)
    else:
        if rank == 0:
            return _kskipmrr_gpu(A, b, epsilon, k, T, pu)
        else:
            _kskipmrr_gpu(A, b, epsilon, k, T, pu)
