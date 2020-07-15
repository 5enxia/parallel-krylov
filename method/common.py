def _start(method_name='', k=None):
    """[summary]

    Args:
        method_name (str, optional): [手法名]. Defaults to ''.
        k ([int], optional): [kの値]. Defaults to None.
    """
    print('# ============== INFO ================= #')
    print(f'Method:\t{ method_name }')
    print(f'initial_k:\t{ k }')


def _end(
    elapsed_time: float,
    isConverged: bool,
    num_of_iter: int,
    residual,
    residual_index,
    final_k=None
):
    """[summary]

    Args:
        elapsed_time (float): [経過時間]
        isConverged (bool): [収束判定]
        num_of_iter (int): [反復回数(k段飛ばしの場合はk+1反復毎に1回実行)]
        residual ([type]): [残差履歴]
        residual_index ([type]): [収束した時の残差のインデックス]
        final_k ([type], optional): [Adaptiveを実行した際の最終的なk]. Defaults to None.
    """
    print(f'time:\t{ elapsed_time } s')
    status = 'converged' if isConverged else 'diverged'
    print(f'status:\t{ status }')
    if isConverged:
        print(f'iteration:\t{ num_of_iter } times')
        print(f'residual:\t{residual[residual_index]}')
        if final_k:
            print(f'final k:\t{final_k}')
    print('# ===================================== #')
