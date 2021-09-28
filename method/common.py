class Color:
    BLACK = '\033[30m'
    RED = '\033[31m'
    GREEN = '\033[32m'
    YELLOW = '\033[33m'
    BLUE = '\033[34m'
    PURPLE = '\033[35m'
    CYAN = '\033[36m'
    WHITE = '\033[37m'
    END = '\033[0m'
    BOLD = '\038[1m'
    UNDERLINE = '\033[4m'
    INVISIBLE = '\033[08m'
    REVERCE = '\033[07m'


def _start(method_name: str = '', k: int = None) -> None:
    """[summary]

    Args:
        method_name (str, optional): [description]. Defaults to ''.
        k (int, optional): [description]. Defaults to None.
    """
    print(Color.GREEN, '# ', '='*16, ' INFO ', '='*16, ' #', Color.END, sep='')
    print(f'Method:\t\t{ method_name }')
    print(f'initial_k:\t{ k }')


def _end(
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
    print(f'time:\t\t{ elapsed_time } s')
    status = 'converged' if isConverged else 'diverged'
    print(f'status:\t\t{ status }')
    print(f'iteration:\t{ num_of_iter } times')
    print(f'final residual:\t{ final_residual }')
    if final_k:
        print(f'final k:\t{final_k}')
    print(Color.GREEN, '# ', '='*38, ' #', Color.END, sep='')
