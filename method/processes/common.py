import numpy as np
from numpy.linalg import norm
from mpi4py import MPI

from ..common import _start, _end


def start(method_name='', k=None):
    """計算開始時処理
    Args:
        method_name (str, optional): 手法名 Defaults to ''.
        k (int, optional): k. Defaults to None.

    Returns:
        float: 計算開始時刻
    """
    _start(method_name, k)
    return MPI.Wtime()


def end(start_time, isConverged, num_of_iter, final_residual, final_k=None):
    """計算終了処理

    Args:
        start_time (float): 計算開始時刻
        isConverged (bool): 収束判定
        num_of_iter (int): 反復回数
        final_residual (numpy.ndarray): 最終残差
        residual_index (int): 反復終了時の残差インデックス
        final_k (int, optional): 反復終了時のk Defaults to None.

    Returns:
        float: 経過時間
    """
    elapsed_time = MPI.Wtime() - start_time
    _end(elapsed_time, isConverged, num_of_iter, final_residual, final_k)
    return elapsed_time


def _init(A, b, num_of_process, T=np.float64):
    """[summary]
    クリロフ部分空間法に共通する変数を初期化して返す

    Args:
        A (np.ndarray): 係数行列
        b (np.ndarray): 右辺ベクトル
        num_of_process (int): MPIプロセス数
        T (dtype, optional): 浮動小数精度. Defaults to np.float64.

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
    old_N = b.size
    num_of_append = ((num_of_process - (old_N % num_of_process)) % num_of_process)
    N = old_N + num_of_append
    local_N = N // num_of_process

    if num_of_append:
        A = np.append(A, np.zeros((old_N, num_of_append)), axis=1)  # 右に0を追加
        A = np.append(A, np.zeros((num_of_append, N)), axis=0)  # 下に0を追加
        b = np.append(b, np.zeros(num_of_append))  # 0を追加
    x = np.zeros(N, T)
    b_norm = norm(b)

    max_iter = old_N * 2
    residual = np.zeros(max_iter+1, T)
    num_of_solution_updates = np.zeros(max_iter+1, np.int)
    num_of_solution_updates[0] = 0
    return A, b, x, b_norm, N, local_N, max_iter, residual, num_of_solution_updates
