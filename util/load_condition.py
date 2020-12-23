import json

import numpy as np


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

        return epsilon, N, diag, sub_diag, k 
