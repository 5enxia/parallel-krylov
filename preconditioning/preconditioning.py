import numpy as np
from scipy.sparse import linalg,lil_matrix,csc_matrix,diags

# LU decomposition
def lu(A):
    csc_A = lil_matrix(A).tocsc()
    lu = linalg.splu(csc_A,permc_spec='NATURAL')
    L = lu.L.toarray()
    U = lu.U.toarray()
    return np.linalg.inv(L).dot(np.linalg.inv(U))

# ncomplete LU decomposition
def ilu(A):
    csc_A = lil_matrix(A).tocsc()
    ilu = linalg.spilu(csc_A)
    L = ilu.L.toarray()
    U = ilu.U.toarray()
    return np.linalg.inv(L).dot(np.linalg.inv(U))

# Diagonal Scaling
def diagScaling(A):
    return np.diagflat(A.diagonal())