import json

import numpy as np


def load_matrix(path: str, version: str, T=np.float64):
    """Meshless-Matrixのmatrixからnumpy.ndarray型のデータを生成

    Args:
        path (str): データファイルパス
        version (str): 'EFG' or 'reduced'
        T ([type]): 浮動小数精度

    Returns:
        numpy.ndarray: 係数行列
    """
    with open(path, 'r') as f:
        arr = []
        num = 0
        for line in f:
            if num == 0:
                size = line.split()
            else:
                arr.append(float(line))
            num += 1

        if version == 'EFG':
            size = int(size[0]) + int(size[1])
        else:
            size = int(size[0])

        return np.array(arr, T).reshape((size, size))


def load_vector(path: str, version: str, T=np.float64):
    """Meshless-Matrixのvectorからnumpy.ndarray型のデータを生成

    Args:
        path (str): データファイルパス
        version (str): 'EFG' or 'reduced'
        T (numpy.dtype): 浮動小数精度

    Returns:
        numpy.ndarray: ベクトル
    """
    with open(path, 'r') as f:
        arr = []
        num = 0
        for line in f:
            if num == 0:
                pass
            else:
                arr.append(float(line))
            num += 1

        return np.array(arr, T)


def load_condition_params(path: str, T=np.float64):
    """condition.jsonから数値条件を読み込む

    Args:
        path (str): ファイルパス
        T ([numpy.dtype], optional): 浮動小数精度

    Returns:
        float: 収束判定子
        int: 次元数
        float: 第一対角要素
        float: 第二対角要素
        int: k
        numpy.dtype: 浮動小数精度
    """
    with open(path) as f:
        params = json.load(f)

        epsilon = params['epsilon']
        N = params['N']
        diag = params['diag']
        sub_diag = params['sub_diag']
        k = params['k']

        return epsilon, N, diag, sub_diag, k, T
