import numpy as np
from mpi4py import MPI

from ..common import _init


def init(A, b, num_of_process, k=0, T=np.float64):
    """[summary]
    クリロフ部分空間法に共通する変数を初期化して返す

    Args:
        A (np.ndarray): 係数行列
        b (np.ndarray): 右辺ベクトル
        num_of_process (int): MPIプロセス数
        T (np.float64, optional): 浮動小数精度. Defaults to np.float64.

    Returns:
        A (np.ndarray): 係数行列(0パッディング)
        b (np.ndarray): 右辺ベクトル(0パッディング)
        x (np.ndarray): 初期解
        b_norm (float): bのL2ノルム
        N (int): パッディング後の次元数
        local_N (int): ローカル行列の縦方向次元数
        max_iter (int): 最大反復回数（パッディング前の次元数）
        residual (np.ndarray): 残差履歴
        num_of_solution_updates (np.ndarray): 残差更新回数履歴
    """
    return _init(A, b, num_of_process, T)


def init_mpi():
    """[summary]

    Returns:
        [MPI.COMM_WORLD]: [MPI通信範囲]
        [int]: [ランク]
        [int]: [総プロセス数]
    """
    comm = MPI.COMM_WORLD
    return comm, comm.Get_rank(), comm.Get_size()


def init_matvec(N, local_N, T=np.float64):
    """[summary]
    mpi_matvecを実行する際に必要なローカル変数を初期化して返す

    Args:
        N ([type]): [description]
        local_N ([type]): [description]
        T ([type], optional): [description]. Defaults to np.float64.

    Returns:
        [np.ndarray]: [ローカル係数行列]
        [np.ndarray]: [A.dot(x)の演算結果を格納する使い回しベクトル(N)]
        [np.ndarray]: [local_A.dot(x)の演算結果を格納する使い回しベクトル(local_N)]
    """
    local_A = np.empty((local_N, N), T)
    Ax = np.empty(N, T)
    local_Ax = np.empty(local_N, T)
    return local_A, Ax, local_Ax


def init_vecvec(local_N, T=np.float64):
    """[summary]
    mpi_vecvecを実行する際に必要なローカル変数を初期化して返す

    Args:
        local_N ([int]): [ローカル次元数]
        T ([np.dtype], optional): [numpy数値精度]. Defaults to np.float64.

    Returns:
        [np.ndarray]: [演算結果を格納する使い回しベクトル(local_N)]
        [np.ndarray]: [演算結果を格納する使い回しベクトル(local_N)]
    """
    local_a = np.empty(local_N, T)
    local_b = np.empty(local_N, T)
    return local_a, local_b


def mpi_matvec(local_A, x, Ax, local_Ax, comm):
    """[summary]
    returnを使う（計算式の中でmpi_matvecを使う）場合は，Axとlocal_Axを使う．
    returnを使わない場合は，Ax, local_Axには，対象の配列をいれる．

    Args:
        local_A ([type]): [ローカル行列(local_N * N)]
        x (np.ndarray): [ベクトル]
        Ax (np.ndarray): [A.dot(x)の結果を格納]
        local_Ax (np.ndarray): [local_A.dot(x)の結果を格納]
        comm (): [MPI.COMM_WORLD()]

    Returns:
        Ax [np.ndarray]: [演算結果]
    """
    comm.Bcast(x, root=0)
    local_Ax = np.dot(local_A, x)
    comm.Gather(local_Ax, Ax, root=0)
    return Ax


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
        ab [np.ndarray]: [演算結果]
    """
    ab = np.empty(1, np.float64)
    comm.Scatter(a, local_a, root=0)
    comm.Scatter(b, local_b, root=0)
    local_ab = np.dot(local_a, local_b)
    comm.Reduce(local_ab, ab, root=0)
    return ab
