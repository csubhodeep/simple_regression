import unittest
import numpy as np
from hour_regression import HourRegression

class TestHourRegressionMethods(unittest.TestCase):

    def setUp(self):
        self.test_object = HourRegression()

    def test_adjusted_r2_score(self):
        a1 = np.array([1,2,3,4])
        b1 = np.array([1,2,3,4])
        p = 1
        self.assertEqual(self.test_object.adjusted_r2_score(a1,b1,p), 1.0)
        a2 = np.array([1,2,3,4,5])
        self.assertRaises(AssertionError,self.test_object.adjusted_r2_score(a2,b1,p))

    def test_main(self):
        path = ""
        self.assertRaises(AssertionError,self.test_object.main(path))

    def test_pre_process(self):
        data = "not a numpy float array"
        self.assertRaises(AssertionError,self.test_object.pre_process(data))

    def tearDown(self):
        self.test_object.dispose()


if __name__ == '__main__':
    unittest.main()