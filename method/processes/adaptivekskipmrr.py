import numpy as np

from .common import start, end, init, init_mpi


def _adaptivekskipmrr_cpu(A, b, epsilon, k, T, pu):
    comm, rank, num_of_process = init_mpi()
    import numpy as xp
    from numpy import dot
    from numpy.linalg import norm
    from .cpu import init_matvec, init_vecvec, mpi_matvec, mpi_vecvec1, mpi_vecvec2

    # 共通初期化
    A, b, x, b_norm, N, local_N, max_iter, residual, num_of_solution_updates = init(A, b, num_of_process, T, pu)
    local_A, Ax, local_Ax = init_matvec(N, local_N, T)
    local_a, local_b = init_vecvec(local_N, T)
    comm.Scatter(A, local_A)
    # root
    Ar = xp.empty((k + 3, N), T)
    Ay = xp.empty((k + 2, N), T)
    alpha = xp.empty(2*k + 3, T)
    beta = xp.empty(2*k + 2, T)
    beta[0] = 0
    delta = xp.empty(2*k + 1, T)
    # local
    local_alpha = xp.empty(2*k + 3, T)
    local_beta = xp.empty(2*k + 2, T)
    local_beta[0] = 0
    local_delta = xp.empty(2*k + 1, T)
    
    k_history = np.zeros(max_iter+1, np.int)
    k_history[0] = k

    # 初期残差
    Ar[0] = b - mpi_matvec(local_A, x, Ax, local_Ax, comm)
    residual[0] = norm(Ar[0]) / b_norm
    cur_residual = residual[0]
    pre_residual = residual[0]

    # 初期反復
    if rank == 0:
        start_time = start(method_name='adaptive k-skip MrR', k=k)
    Ar[1] = mpi_matvec(local_A, Ar[0], Ax, local_Ax, comm)
    zeta = mpi_vecvec2(Ar[0], Ar[1], local_a, local_b, comm) / mpi_vecvec1(Ar[1], local_a, comm)
    Ay[0] = zeta * Ar[1]
    z = -zeta * Ar[0]
    Ar[0] -= Ay[0]
    x -= z
    num_of_solution_updates[1] = 1
    k_history[1] = k
    i = 1
    index = 1

    # 反復計算
    while i < max_iter:
        pre_residual = cur_residual
        cur_residual = norm(Ar[0]) / b_norm
        residual[index] = cur_residual
        # 残差減少判定
        isIncreaese = np.array([cur_residual > pre_residual], bool)
        comm.Bcast(isIncreaese, root=0)
        if isIncreaese:
            # 解と残差を再計算
            x = pre_x.copy()
            Ar[0] = b - mpi_matvec(local_A, x, Ax, local_Ax, comm)
            Ar[1] = mpi_matvec(local_A, Ar[0], Ax, local_Ax, comm)
            zeta = mpi_vecvec2(Ar[0], Ar[1], local_a, local_b, comm) / mpi_vecvec1(Ar[1], local_a, comm)
            Ay[0] = zeta * Ar[1]
            z = -zeta * Ar[0]
            Ar[0] -= Ay[0]
            x -= z

            i += 1
            index += 1
            num_of_solution_updates[index] = i
            residual[index] = norm(Ar[0]) / b_norm

            # kを下げて収束を安定化させる
            if k > 1:
                k -= 1
            k_history[index] = k
        else:
            pre_x = x.copy()
            
        # 収束判定
        isConverged = np.array([cur_residual < epsilon], dtype=bool)
        comm.Bcast(isConverged, root=0)
        if isConverged:
            break

        # 事前計算
        for j in range(1, k + 2):
            Ar[j] = mpi_matvec(local_A, Ar[j-1], Ax, local_Ax, comm)
        for j in range(1, k + 1):
            Ay[j] = mpi_matvec(local_A, Ay[j-1], Ax, local_Ax, comm)
        comm.Bcast(Ar)
        comm.Bcast(Ay)
        for j in range(2*k + 3):
            jj = j // 2
            local_alpha[j] = dot(
                Ar[jj][rank * local_N: (rank+1) * local_N],
                Ar[jj + j % 2][rank * local_N: (rank+1) * local_N]
            )
        comm.Reduce(local_alpha, alpha)
        for j in range(1, 2 * k + 2):
            jj = j//2
            local_beta[j] = dot(
                Ay[jj][rank * local_N: (rank+1) * local_N],
                Ar[jj + j % 2][rank * local_N: (rank+1) * local_N]
            )
        comm.Reduce(local_beta, beta)
        for j in range(2 * k + 1):
            jj = j // 2
            local_delta[j] = dot(
                Ay[jj][rank * local_N: (rank+1) * local_N],
                Ay[jj + j % 2][rank * local_N: (rank+1) * local_N]
            )
        comm.Reduce(local_delta, delta)

        # MrRでの1反復(解と残差の更新)
        d = alpha[2] * delta[0] - beta[1] ** 2
        zeta = alpha[1] * delta[0] / d
        eta = -alpha[1] * beta[1] / d
        Ay[0] = eta * Ay[0] + zeta * Ar[1]
        z = eta * z - zeta * Ar[0]
        Ar[0] -= Ay[0]
        Ar[1] = mpi_matvec(local_A, Ar[0], Ax, local_Ax, comm)
        x -= z

        # MrRでのk反復
        for j in range(0, k):
            delta[0] = zeta ** 2 * alpha[2] + eta * zeta * beta[1]
            alpha[0] -= zeta * alpha[1]
            delta[1] = eta ** 2 * delta[1] + 2 * eta * zeta * beta[2] + zeta ** 2 * alpha[3]
            beta[1] = eta * beta[1] + zeta * alpha[2] - delta[1]
            alpha[1] = -beta[1]
            for l in range(2, 2 * (k - j) + 1):
                delta[l] = eta ** 2 * delta[l] + 2 * eta * zeta * beta[l + 1] + zeta ** 2 * alpha[l + 2]
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
            Ar[1] = mpi_matvec(local_A, Ar[0], Ax, local_Ax, comm)
            x -= z

        i += (k + 1)
        index += 1
        num_of_solution_updates[index] = i
        k_history[index] = k
    else:
        isConverged = False
        residual[index] = norm(Ar[0]) / b_norm

    num_of_iter = i
    if rank == 0:
        elapsed_time = end(start_time, isConverged, num_of_iter, residual[index], k)
        return elapsed_time, num_of_solution_updates[:index+1], residual[:index+1], k_history[:index+1]
    else:
        exit(0)


def _adaptivekskipmrr_gpu(A, b, epsilon, k, T, pu):
    comm, rank, num_of_process = init_mpi()
    import cupy as xp
    from cupy import dot
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
    local_A = xp.asarray(local_A_cpu)
    # root
    Ar = xp.empty((k + 3, N), T)
    Ay = xp.empty((k + 2, N), T)
    alpha = xp.empty(2*k + 3, T)
    beta = xp.empty(2*k + 2, T)
    beta[0] = 0
    delta = xp.empty(2*k + 1, T)
    # local
    local_alpha = xp.empty(2*k + 3, T)
    local_beta = xp.empty(2*k + 2, T)
    local_beta[0] = 0
    local_delta = xp.empty(2*k + 1, T)
    # root_cpu
    Ar_cpu = np.empty((k + 3, N), T)
    Ay_cpu = np.empty((k + 2, N), T)
    alpha_cpu = np.empty(2*k + 3, T)
    beta_cpu = np.empty(2*k + 2, T)
    beta[0] = 0
    delta_cpu = np.empty(2*k + 1, T)
    
    k_history = np.zeros(max_iter+1, np.int)
    k_history[0] = k

    # 初期残差
    Ar[0] = b - mpi_matvec(local_A, x, Ax, local_Ax, comm)
    residual[0] = norm(Ar[0]) / b_norm
    cur_residual = residual[0]
    pre_residual = residual[0]

    # 初期反復
    if rank == 0:
        start_time = start(method_name='adaptive k-skip MrR', k=k)
    Ar[1] = mpi_matvec(local_A, Ar[0], Ax, local_Ax, comm)
    zeta = mpi_vecvec2(Ar[0], Ar[1], local_a, local_b, comm) / mpi_vecvec1(Ar[1], local_a, comm)
    Ay[0] = zeta * Ar[1]
    z = -zeta * Ar[0]
    Ar[0] -= Ay[0]
    x -= z
    num_of_solution_updates[1] = 1
    k_history[1] = k
    i = 1
    index = 1

    # 反復計算
    while i < max_iter:
        pre_residual = cur_residual
        cur_residual = norm(Ar[0]) / b_norm
        residual[index] = cur_residual
        # 残差減少判定
        isIncreaese = np.array([cur_residual > pre_residual], bool)
        comm.Bcast(isIncreaese, root=0)
        if isIncreaese:
            # 解と残差を再計算
            x = pre_x.copy()
            Ar[0] = b - mpi_matvec(local_A, x, Ax, local_Ax, comm)
            Ar[1] = mpi_matvec(local_A, Ar[0], Ax, local_Ax, comm)
            zeta = mpi_vecvec2(Ar[0], Ar[1], local_a, local_b, comm) / mpi_vecvec1(Ar[1], local_a, comm)
            Ay[0] = zeta * Ar[1]
            z = -zeta * Ar[0]
            Ar[0] -= Ay[0]
            x -= z

            i += 1
            index += 1
            num_of_solution_updates[index] = i
            residual[index] = norm(Ar[0]) / b_norm

            # kを下げて収束を安定化させる
            if k > 1:
                k -= 1
            k_history[index] = k
        else:
            pre_x = x.copy()
            
        # 収束判定
        isConverged = np.array([cur_residual < epsilon], dtype=bool)
        comm.Bcast(isConverged, root=0)
        if isConverged:
            break

        # 事前計算
        for j in range(1, k + 2):
            Ar[j] = mpi_matvec(local_A, Ar[j-1], Ax, local_Ax, comm)
        for j in range(1, k + 1):
            Ay[j] = mpi_matvec(local_A, Ay[j-1], Ax, local_Ax, comm)
        Ar_cpu = Ar.get()
        Ay_cpu = Ay.get()
        comm.Bcast(Ar_cpu)
        comm.Bcast(Ay_cpu)
        Ar = xp.asarray(Ar_cpu)
        Ay = xp.asarray(Ay_cpu)
        for j in range(2*k + 3):
            jj = j // 2
            local_alpha[j] = dot(
                Ar[jj][rank * local_N: (rank+1) * local_N],
                Ar[jj + j % 2][rank * local_N: (rank+1) * local_N]
            )
        comm.Reduce(local_alpha.get(), alpha_cpu, root=0)
        alpha = xp.asarray(alpha_cpu)
        for j in range(1, 2 * k + 2):
            jj = j//2
            local_beta[j] = dot(
                Ay[jj][rank * local_N: (rank+1) * local_N],
                Ar[jj + j % 2][rank * local_N: (rank+1) * local_N]
            )
        comm.Reduce(local_beta.get(), beta_cpu, root=0)
        beta = xp.asarray(beta_cpu)
        for j in range(2 * k + 1):
            jj = j // 2
            local_delta[j] = dot(
                Ay[jj][rank * local_N: (rank+1) * local_N],
                Ay[jj + j % 2][rank * local_N: (rank+1) * local_N]
            )
        comm.Reduce(local_delta.get(), delta_cpu, root=0)
        delta = xp.asarray(delta_cpu)

        # MrRでの1反復(解と残差の更新)
        d = alpha[2] * delta[0] - beta[1] ** 2
        zeta = alpha[1] * delta[0] / d
        eta = -alpha[1] * beta[1] / d
        Ay[0] = eta * Ay[0] + zeta * Ar[1]
        z = eta * z - zeta * Ar[0]
        Ar[0] -= Ay[0]
        Ar[1] = mpi_matvec(local_A, Ar[0], Ax, local_Ax, comm)
        x -= z

        # MrRでのk反復
        for j in range(0, k):
            delta[0] = zeta ** 2 * alpha[2] + eta * zeta * beta[1]
            alpha[0] -= zeta * alpha[1]
            delta[1] = eta ** 2 * delta[1] + 2 * eta * zeta * beta[2] + zeta ** 2 * alpha[3]
            beta[1] = eta * beta[1] + zeta * alpha[2] - delta[1]
            alpha[1] = -beta[1]
            for l in range(2, 2 * (k - j) + 1):
                delta[l] = eta ** 2 * delta[l] + 2 * eta * zeta * beta[l + 1] + zeta ** 2 * alpha[l + 2]
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
            Ar[1] = mpi_matvec(local_A, Ar[0], Ax, local_Ax, comm)
            x -= z

        i += (k + 1)
        index += 1
        num_of_solution_updates[index] = i
        k_history[index] = k
    else:
        isConverged = False
        residual[index] = norm(Ar[0]) / b_norm

    num_of_iter = i
    if rank == 0:
        elapsed_time = end(start_time, isConverged, num_of_iter, residual[index], k)
        return elapsed_time, num_of_solution_updates[:index+1], residual[:index+1], k_history[:index+1]
    else:
        exit(0)


def adaptivekskipmrr(A, b, epsilon, k, T, pu):
    comm, rank, num_of_process = init_mpi()
    if pu == 'cpu':
        if rank == 0:
            return _adaptivekskipmrr_cpu(A, b, epsilon, k, T, pu)
        else:
            _adaptivekskipmrr_cpu(A, b, epsilon, k, T, pu)
    else:
        if rank == 0:
            return _adaptivekskipmrr_gpu(A, b, epsilon, k, T, pu)
        else:
            _adaptivekskipmrr_gpu(A, b, epsilon, k, T, pu)