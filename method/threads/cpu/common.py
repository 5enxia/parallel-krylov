import numpy as np

from ..common import _init


def init(A, b, T=np.float64):
    """[summary]

    Args:
        A ([type]): [係数行列]
        b ([type]): [厳密解]
        T ([type], optional): [description]. Defaults to np.float64.

    Returns:
        x [np.zeros]: [初期解]
        b_norm [float64]: [bのL2ノルム]
        N [int]: [次元数]
        max_iter [int]: [最大反復回数]
        residual [np.zeros]: [残差履歴]
        num_of_solution_updates [np.zeros]: [解の更新回数履歴]
    """
    return _init(A, b, T)
