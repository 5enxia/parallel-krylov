import argparse
import numpy as np
import os


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


parser = argparse.ArgumentParser(description='.txt to .npy converter')
parser.add_argument("path", help="data file path")
parser.add_argument("-t", "--type", help="vector file")
parser.add_argument("-m", "--matrix", action="store_true",
                    help="matrix file")
parser.add_argument("-v", "--vector", action="store_true",
                    help="vector file")

args = parser.parse_args()
path, basename = os.path.split(args.path)
filename, ext = os.path.splitext(basename)
if args.matrix:
    matrix = load_matrix(args.path, args.type)
    np.save(f'{path}/{filename}.npy', matrix)
elif args.vector:
    vector = load_vector(args.path, args.type)
    np.save(f'{path}/{filename}.npy', vector)
