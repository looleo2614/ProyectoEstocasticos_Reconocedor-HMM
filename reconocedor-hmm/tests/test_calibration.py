import unittest
import numpy as np
from src.HMMEngine import HMM, classify
from scripts.calibrate import nll

class CalibrationTests(unittest.TestCase):
    def test_temperature_one_is_default(self):
        a = HMM(np.array([[1.]]), np.array([[.8, .2]]), np.array([1.]))
        b = HMM(np.array([[1.]]), np.array([[.3, .7]]), np.array([1.]))
        models = {'a': a, 'b': b}
        self.assertEqual(classify(models, [0]), classify(models, [0], temperature=1.))
    def test_temperature_softens_distribution(self):
        a = HMM(np.array([[1.]]), np.array([[.8, .2]]), np.array([1.]))
        b = HMM(np.array([[1.]]), np.array([[.3, .7]]), np.array([1.]))
        models = {'a': a, 'b': b}
        self.assertLess(classify(models, [0], temperature=3.)[0][1], classify(models, [0])[0][1])
    def test_invalid_temperature(self):
        with self.assertRaises(ValueError): classify({'a': HMM([[1]], [[1]], [1])}, [0], temperature=0)
    def test_calibration_nll(self):
        rows = [{'label': 'a', 'scores': {'a': -1., 'b': -3.}}]
        self.assertLess(nll(rows, 1.), nll(rows, 5.))

if __name__ == '__main__': unittest.main()
