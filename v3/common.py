# 標準出力
def _start(method_name: str = '', k: int = None) -> None:
    """[summary]

    Args:
        method_name (str, optional): [description]. Defaults to ''.
        k (int, optional): [description]. Defaults to None.
    """
    print('# ', '='*16, ' INFO ', '='*16, ' #', sep='')
    print(f'Method:\t\t{ method_name }')
    if k != None:
        print(f'Initial_k:\t{ k }')


def _finish(
    elapsed_time: float,
    isConverged: bool,
    num_of_iter: int,
    final_residual: float,
    final_k: int = None
) -> None:
    """[summary]

    Args:
        elapsed_time (float): [description]
        isConverged (bool): [description]
        num_of_iter (int): [description]
        final_residual (float): [description]
        final_k (int, optional): [description]. Defaults to None.
    """
    print(f'Time:\t\t{ elapsed_time } s')
    status = 'converged' if isConverged else 'diverged'
    print(f'Status:\t\t{ status }')
    print(f'Iteration:\t{ num_of_iter } times')
    print(f'Final_Residual:\t{ final_residual }')
    if final_k:
        print(f'Final_k:\t{final_k}')
    print('# ', '='*38, ' #', sep='')
