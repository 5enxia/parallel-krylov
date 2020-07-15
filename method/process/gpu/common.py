import numpy as np
import cupy as cp
from cupy.linalg import norm


def init(A, b, T=cp.float64):
    """[summary]
    クリロフ部分空間法に共通する変数を初期化して返す

    Args:
        A ([type]): [description]
        b ([type]): [description]
        T ([type], optional): [description]. Defaults to np.float64.

    Returns:
        [type]: [description]
    """
    x = cp.zeros(b.size, T)
    b_norm = norm(b)
    N = b.size
    max_iter = N  # * 2
    residual = cp.zeros(max_iter+1, T)
    num_of_solution_updates = cp.zeros(max_iter+1, cp.int)
    num_of_solution_updates[0] = 0
    return x, b_norm, N, max_iter, residual, num_of_solution_updates


def init_matvec(N, num_of_process, T=np.float64):
    """[summary]
    mpi_matvecを実行する際に必要なローカル変数を初期化して返す

    Args:
        N ([type]): [description]
        num_of_process ([type]): [description]
        T ([type], optional): [description]. Defaults to np.float64.

    Returns:
        [type]: [description]
    """
    local_N = N // num_of_process
    local_A = cp.empty((local_N, N), T)
    Ax = np.empty(N, T)
    local_Ax = cp.empty(local_N, T)
    return local_N, local_A, Ax, local_Ax


def init_vecvec(local_N, T=cp.float64):
    """[summary]
    mpi_vecvecを実行する際に必要なローカル変数を初期化して返す

    Args:
        local_N ([type]): [description]
        T ([type], optional): [description]. Defaults to np.float64.

    Returns:
        [type]: [description]
    """
    local_a = cp.empty(local_N, T)
    local_b = cp.empty(local_N, T)
    return local_a, local_b


def mpi_matvec(local_A, x, Ax, local_Ax, comm):
    """[summary]
    returnを使う（計算式の中でmpi_matvecを使う）場合は，Axとlocal_Axを使う．
    returnを使わない場合は，Ax, local_Axには，対象の配列をいれる．

    Args:
        local_A (cp.ndarray): [ローカル行列(local_N * N)]
        x (cp.ndarray): [ベクトル]
        Ax (np.ndarray): [A.dot(x)の結果を格納]
        local_Ax (cp.ndarray): [local_A.dot(x)の結果を格納]
        comm (): [MPI.COMM_WORLD()]

    Returns:
        [cp.ndarray]: [演算結果]
    """
    x_cpu = x.get()
    comm.Bcast(x_cpu, root=0)
    x = cp.asarray(x_cpu)

    local_Ax = cp.dot(local_A, x)

    comm.Gather(local_Ax.get(), Ax, root=0)
    return cp.asarray(Ax)


def mpi_vecvec(a, b, local_a, local_b, comm):
    """[summary]
    returnを使う（計算式の中でmpi_matvecを使う）場合は，abとlocal_abを使う．
    returnを使わない場合は，ab, local_abには，対象の配列をいれる．

    Args:
        a (np.ndarray): [ベクトル1]
        b ([type]): [ベクトル2]
        local_a ([type]): [ローカルベクトル1]
        local_b ([type]): [ローカルベクトル2]
        comm (): [MPI.COMM_WORLD()]

    Returns:
        [np.ndarray]: [演算結果]
    """
    ab = np.empty(1, cp.float64)
    local_a_cpu = local_a.get()
    local_b_cpu = local_b.get()
    comm.Scatter(a.get(), local_a_cpu, root=0)
    comm.Scatter(b.get(), local_b_cpu, root=0)
    local_a = cp.asarray(local_a_cpu)
    local_b = cp.asarray(local_b_cpu)
    
    local_ab = cp.dot(local_a, local_b)

    comm.Reduce(local_ab.get(), ab, root=0)
    return cp.asarray(ab)
