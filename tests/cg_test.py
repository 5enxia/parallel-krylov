import sys
import unittest

sys.path.append('../../')
from krylov.util import loader
from krylov.single.cg import cg
import numpy as np 

class TestCgMethod(unittest.TestCase):
    # COMMON Setting
    epsilon = 1e-8
    T = np.float64
    directory = '../data/'

    # test 1
    version, N, ext = 'EFG', 1081, '.txt'
    A = loader.matrixLoader(f'{directory}matrix_{version}-{N}{ext}', version, N, T)
    b = loader.vectorLoader(f'{directory}vector_{version}-{N}{ext}', version, N, T)

    def test_single_cg_method_with_EFG_1081(self):
        self.assertTrue(cg(TestCgMethod.A, TestCgMethod.b, TestCgMethod.epsilon, TestCgMethod.T))

if __name__ == "__main__":
    unittest.main()