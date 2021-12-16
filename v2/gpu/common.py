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
    if num_of_append:
        b = np.append(b, np.zeros(num_of_append))  # 0を追加
    b_norm = np.linalg.norm(b)

    # x
    x: np.ndarray = np.zeros(N, T)

    # その他パラメータ
    max_iter: int = old_N * 2
    residual: np.ndarray = np.zeros(max_iter+1, T)
    num_of_solution_updates: np.ndarray = np.zeros(max_iter+1, np.int)
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
        MultiGpu.begin = begin
        MultiGpu.end = end
        MultiGpu.num_of_gpu = end - begin + 1

        # init memory allocator
        for i in range(end, begin-1, -1):
            Device(i).use()
            pool = cp.cuda.MemoryPool(cp.cuda.malloc_managed)
            cp.cuda.set_allocator(pool.malloc)

    
    # メモリー領域を確保
    @classmethod
    def alloc(cls, A, b, T):
        # dimentional size
        MultiGpu.N = b.size
        MultiGpu.local_N = MultiGpu.N // MultiGpu.num_of_gpu
        # byte size
        MultiGpu.nbytes = b.nbytes
        MultiGpu.local_nbytes = b.nbytes // MultiGpu.num_of_gpu

        # init list
        MultiGpu.A = [None] * MultiGpu.num_of_gpu
        MultiGpu.x = [None] * MultiGpu.num_of_gpu
        MultiGpu.y = [None] * MultiGpu.num_of_gpu

        # divide single A -> multi local_A
        # allocate x, y
        for i in range(MultiGpu.end, MultiGpu.begin-1, -1):
            Device(i).use()
            MultiGpu.A[i-MultiGpu.begin] = cp.array(A[i*MultiGpu.local_N:(i+1)*MultiGpu.local_N]) # Note: Change line when use csr
            MultiGpu.x[i-MultiGpu.begin] = cp.array(MultiGpu.N, T)
            MultiGpu.y[i-MultiGpu.begin] = cp.array(MultiGpu.local_N, T)

        # init out vector
        MultiGpu.out = np.empty(MultiGpu.N)

    # マルチGPUを用いた行列ベクトル積
    @classmethod
    def dot(cls, A, x):
        # Copy vector data to All devices
        for i in range(MultiGpu.end, MultiGpu.begin-1, -1):
            Device(i).use()
            cp.cuda.runtime.memcpyPeer(MultiGpu.x[i].data.ptr, i, x.data.ptr, 0, MultiGpu.nbytes)
        # dot
        for i in range(MultiGpu.end, MultiGpu.begin-1, -1):
            Device(i).use()
            cp.dot(MultiGpu.A[i-MultiGpu.begin], MultiGpu.x[i], out=MultiGpu.y[i-MultiGpu.begin])
        # Gather caculated element from All devices
        for i in range(MultiGpu.end, MultiGpu.begin-1, -1):
            Device(i).synchronize()
            cp.cuda.runtime.memcpyPeer(MultiGpu.out[MultiGpu.local_N*i].data.ptr, 0, MultiGpu.y[i-MultiGpu.begin].data.ptr, i, MultiGpu.local_nbytes)
        # return
        return MultiGpu.out
