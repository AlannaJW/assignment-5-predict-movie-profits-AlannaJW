import unittest
import numpy as np
from src.data_prep import all_classes_prep, edm_rap_prep
from src.logistic_models import logistic_all_classes, logistic_edm_rap

class TestDataPrep(unittest.TestCase):

    def setUp(self):
        self.f1 = np.array([[15, 19, "t", 80, 72, 93, 29, 20, 10, "rap", 239, 192, 183, 193, 744, 474, 19, 239, 192, 183, 193, 744, 474], [90, 49, "w", 17, 72, 93, 29, 20, 10, "rap", 93, 29, 20, 10, 19, 239, 192, 183, 193, 744, 474, 19, 239], [10, 70, 14, 88, 72, 93, 29, 20, 10, "rap", 93, 29, 20, 10, 19, 239, 192, 183, 193, 744, 474, 19, 239]])
        self.f2 = np.array([[15, 19, 19, 80, 72, 93, 29, 20, 10, "rap", 239, 192, 183, 193, 744, 474, 19, 239, 192, 183, 193, 744, 474], [90, 49, 82, 17, 72, 93, 29, 20, 10, "rap", 93, 29, 20, 10, 19, 239, 192, 183, 193, 744, 474, 19, 239], [10, 70, 14, 88, 72, 93, 29, 20, 10, "rap", 93, 29, 20, 10, 19, 239, 192, 183, 193, 744, 474, 19, 239]])
        self.f3 = np.array([[15, 19, 19, 80, 72, 93, 29, 20, 10, "rap", 239, 192, 183, 193, 744, 474, 19, 239, 92],
                            [90, 49, 82, 17, 72, 93, 29, 20, 10, "rap", 93, 29, 20, 10, 19, 239, 192, 183, 28],
                            [10, 70, 14, 88, 72, 93, 29, 20, 10, "rap", 93, 29, 20, 10, 19, 128, 39, 193, 84]])

    def test_all_classes_prep_AND_edm_rap_prep(self):

        with self.assertRaises(ValueError):
            all_classes_prep(self.f1)
        with self.assertRaises(ValueError):
            all_classes_prep(self.f2)
        with self.assertRaises(ValueError):
            all_classes_prep(self.f3)
        with self.assertRaises(ValueError):
            edm_rap_prep(self.f1)
        with self.assertRaises(ValueError):
            edm_rap_prep(self.f2)
        with self.assertRaises(ValueError):
            edm_rap_prep(self.f3)

# class TestLogisticModels(unittest.TestCase):
#
#     def setUp(self):
#
#
#     def test_logistic_all_classes_AND_logistic_edm_rap(self):
#
#
#     def test_make_y_binary(self):
#
#
#     def test_create_predictions(self):
#
#
#     def test_calc_accuracy(self):


if __name__ == '__main__':
    unittest.main()
