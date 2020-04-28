import sys
import unittest

sys.path.append('../../../')
from krylov.util import loader, toepliz_matrix_generator
from krylov.method.single.mrr import mrr 
import numpy as np 

class TestCgMethod(unittest.TestCase):
    epsilon = 1e-8
    T = np.float64

    def test_single_MrR_method(self):
        A, b = toepliz_matrix_generator.generate(N=1000,diag=2.5)
        self.assertTrue(mrr(A, b, TestCgMethod.epsilon, TestCgMethod.T))

if __name__ == "__main__":
    unittest.main()