import argparse
from scipy.io import mmwrite
from scipy.sparse import load_npz

parser = argparse.ArgumentParser(description='.npz to .npy converter')
parser.add_argument("path", help="data file path")

args = parser.parse_args()

npz = load_npz(args.path)

mmwrite(args.path.replace('npz', 'npy'), npz)
