import numpy as np
from scipy.sparse import linalg, lil_matrix, csc_matrix, diags


def lu(A):
    """LU decomposition

    Args:
        A ([type]): [description]

    Returns:
        [type]: [description]
    """
    csc_A = lil_matrix(A).tocsc()
    lu = linalg.splu(csc_A, permc_spec='NATURAL')
    return lu
    # L = lu.L.toarray()
    # U = lu.U.toarray()
    # return np.linalg.inv(L).dot(np.linalg.inv(U))


def ilu(A):
    """non complete LU decomposition

    Args:
        A ([type]): [description]

    Returns:
        [type]: [description]
    """
    csc_A = lil_matrix(A).tocsc()
    ilu = linalg.spilu(csc_A)
    return ilu
    # L = ilu.L.toarray()
    # U = ilu.U.toarray()
    # return np.linalg.inv(L).dot(np.linalg.inv(U))


def diagScaling(A):
    """Diagonal Scaling

    Args:
        A ([type]): [description]

    Returns:
        [type]: [description]
    """
    return np.diagflat(A.diagonal())

