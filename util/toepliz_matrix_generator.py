import numpy as np
from scipy.linalg import toeplitz


def generate(N: int, diag: float, sub_diag: float, T=np.float64):
    """指定の次元数，対角要素からN次元の行列とベクトルを返す

    Args:
        N (int): 次元数
        diag (float): 第1対角
        sub_diag (float): 第2対角
        T ([type], optional): 浮動小数精度. Defaults to np.float64.

    Returns:
        numpy.ndarray: N*N行列
        numpy.ndarray: Nベクトル
    """
    elements = np.zeros(N, T)
    elements[0] = diag
    elements[1] = sub_diag
    return toeplitz(elements), np.ones(N, T)
