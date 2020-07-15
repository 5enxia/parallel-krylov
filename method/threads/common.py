import time

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
    return time.perf_counter()


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
    elapsed_time = time.perf_counter() - start_time
    _end(elapsed_time, isConverged, num_of_iter, final_residual, final_k)
    return elapsed_time
