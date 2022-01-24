import argparse
from scipy.sparse import load_npz
from numpy import save as save_npy

parser = argparse.ArgumentParser(description='.npz to .npy converter')
parser.add_argument("path", help="data file path")

args = parser.parse_args()

npz = load_npz(args.path)
npy = npz.toarray()

save_npy(args.path.replace('.npz', '.npy'), npy)
