import os
import argparse
from scipy.io import mmread as read
from scipy import sparse

parser = argparse.ArgumentParser(description='.txt to .npy converter')
parser.add_argument("path", help="data file path")

args = parser.parse_args()
path, basename = os.path.split(args.path)
filename, ext = os.path.splitext(basename)

a = read(args.path)
sparse.save(f'{path}/{basename}/.npz', a)
