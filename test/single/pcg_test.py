import sys
import unittest

sys.path.append('../../')
from krylov.util import loader,toepliz_matrix_generator,precondition
from krylov.method.single.pcg import pcg
import numpy as np 

class TestCgMethod(unittest.TestCase):
    epsilon = 1e-4
    T = np.float64

    def test_single_pcg_method(self):
        A, b = toepliz_matrix_generator.generate(N = 2, diag=2.5)
        M = precondition.ilu(A)
        self.assertTrue(pcg(A, b, M ,TestCgMethod.epsilon, TestCgMethod.T))

if __name__ == "__main__":
    unittest.main()