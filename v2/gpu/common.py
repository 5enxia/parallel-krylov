import time

import numpy as np
import scipy
from scipy.sparse import hstack, vstack, csr_matrix
import cupy as cp
from cupy.cuda import Device

from ..common import _start, _finish


# 計測開始
def start(method_name: str = '', k: int = None) -> float:
    """[summary]

    Args:
        method_name (str, optional): [description]. Defaults to ''.
        k (int, optional): [description]. Defaults to None.

    Returns:
        float: [description]
    """
    _start(method_name, k)
    return time.perf_counter()


# 計測終了
def finish(start_time: float, isConverged: bool, num_of_iter: int, final_residual: float, final_k: int = None) -> float:
    """[summary]

    Args:
        start_time (float): [description]
        isConverged (bool): [description]
        num_of_iter (int): [description]
        final_residual (float): [description]
        final_k (int, optional): [description]. Defaults to None.

    Returns:
        float: [description]
    """
    elapsed_time = time.perf_counter() - start_time
    _finish(elapsed_time, isConverged, num_of_iter, final_residual, final_k)
    return elapsed_time

# パラメータの初期化
def init(A, b, T, num_of_thread):
    # 追加する要素数を算出
    old_N = b.size
    num_of_append: int = num_of_thread - (old_N % num_of_thread) # 足りない行を計算
    num_of_append = 0 if num_of_thread == num_of_thread else num_of_append
    N: int = old_N + num_of_append

    ## A
    if num_of_append:
        # データをパディングする
        if isinstance(A, np.ndarray):
            if num_of_append:
                A = np.append(A, np.zeros((old_N, num_of_append)), axis=1)  # 右に0を追加
                A = np.append(A, np.zeros((num_of_append, N)), axis=0)  # 下に0を追加
        elif isinstance(A, scipy.sparse.csr.csr_matrix):
            if num_of_append:
                A = hstack([A, csr_matrix((old_N, num_of_append))], 'csr') # 右にemptyを追加
                A = vstack([A, csr_matrix((num_of_append, N))], 'csr') # 下にemptyを追加

    ## b
    b = cp.array(b, T)
    if num_of_append:
        b = cp.append(b, np.zeros(num_of_append))  # 0を追加
    b_norm = cp.linalg.norm(b)

    # x
    x = cp.zeros(N, T)

    # その他パラメータ
    max_iter = old_N * 2
    residual = cp.zeros(max_iter+1, T)
    num_of_solution_updates = cp.zeros(max_iter+1, np.int)
    num_of_solution_updates[0] = 0

    return A, b, x, b_norm, N, max_iter, residual, num_of_solution_updates


class MultiGpu(object):
    # numbers
    begin: int = 0
    end: int = 0
    num_of_gpu: int = 0
    # dimentinal size
    N: int = 0
    local_N: int = 0
    # matrix
    A: list = []
    # vector
    x: list = []
    y: list = []
    out: np.ndarray = None
    # byte size
    nbytes: int = 0
    local_nbytes: int = 0

    # GPUの初期化
    @classmethod
    def init_gpu(cls, begin: int, end: int):
        cls.begin = begin
        cls.end = end
        cls.num_of_gpu = end - begin + 1

        # init memory allocator
        for i in range(end, begin-1, -1):
            Device(i).use()
            pool = cp.cuda.MemoryPool(cp.cuda.malloc_managed)
            cp.cuda.set_allocator(pool.malloc)

    
    # メモリー領域を確保
    @classmethod
    def alloc(cls, A, b, T):
        # dimentional size
        cls.N = b.size
        cls.local_N = cls.N // cls.num_of_gpu
        # byte size
        cls.nbytes = b.nbytes
        cls.local_nbytes = b.nbytes // cls.num_of_gpu

        # init list
        cls.A = [None] * cls.num_of_gpu
        cls.x = [None] * cls.num_of_gpu
        cls.y = [None] * cls.num_of_gpu

        # divide single A -> multi local_A
        # allocate x, y
        for i in range(cls.end, cls.begin-1, -1):
            Device(i).use()
            index = i-cls.begin
            if isinstance(A, np.ndarray):
                cls.A[index] = cp.array(A[i*cls.local_N:(i+1)*cls.local_N], T) # Note: Change line when use csr
            elif isinstance(A, scipy.sparse.csr.csr_matrix):
                A = csr_matrix(A)
                cls.A[index] = A[i*cls.local_N:(i+1)*cls.local_N] # Note: Change line when use csr
            cls.x[index] = cp.empty(cls.N, T)
            cls.y[index] = cp.empty(cls.local_N, T)

        # init out vector
        cls.out = cp.empty(cls.N, T)

    # マルチGPUを用いた行列ベクトル積
    @classmethod
    def dot(cls, A, x):
        # Copy vector data to All devices
        for i in range(cls.end, cls.begin-1, -1):
            Device(i).use()
            index = i-cls.begin
            cp.cuda.runtime.memcpyPeer(cls.x[index].data.ptr, i, x.data.ptr, 0, cls.nbytes)
        # dot
        for i in range(cls.end, cls.begin-1, -1):
            Device(i).use()
            index = i-cls.begin 
            cls.y[index] = cls.A[index].dot(cls.x[index])
        # Gather caculated element from All devices
        for i in range(cls.end, cls.begin-1, -1):
            Device(i).synchronize()
            index = i-cls.begin
            cp.cuda.runtime.memcpyPeer(cls.out[cls.local_N*i].data.ptr, 0, cls.y[i-cls.begin].data.ptr, i, cls.local_nbytes)
        # return
        return cls.out
