import unittest
import numpy as np
from sklearn.linear_model import LogisticRegression
from src.data_prep import all_classes_prep, edm_rap_prep
from src.logistic_models import logistic_all_classes, logistic_edm_rap, make_y_binary, create_predictions, calc_accuracy

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

class TestLogisticModels(unittest.TestCase):

    def setUp(self):
        self.y_train_1 = ["rap", "edm", "rap", "edm", "rap", "edm", "rap", "edm"]
        self.y_class_data_1 = {"rap" : 0, "edm" : 0}
        self.correct_y_binary = {"rap": [1, 0, 1, 0, 1, 0, 1, 0], "edm" : [0, 1, 0, 1, 0, 1, 0, 1] }

        self.x_test = np.array([[15, 19, 19, 80, 72, 93], [29, 20, 10, 239, 192, 183], [193, 744, 474, 19, 239, 92],
                                [90, 49, 82, 17, 72, 93], [29, 20, 10, 93, 29, 20], [ 10, 19, 239, 192, 183, 28],
                                [10, 70, 14, 88, 72, 93], [29, 20, 10, 93, 29, 20]])
        self.rap_model = LogisticRegression()

        self.rap_model.fit(self.x_test, self.correct_y_binary["rap"])
        self.edm_model = LogisticRegression()
        self.edm_model.fit(self.x_test, self.correct_y_binary["edm"])
        self.class_models = {"rap" : self.rap_model, "edm" : self.edm_model}
        self.clas_pred = {"edm": 0, "rap": 0}
        self.predictions = ['rap', 'edm', 'rap', 'edm', 'rap', 'edm', 'rap', 'rap']

    def test_logistic_all_classes_AND_logistic_edm_rap(self):
        pass

    def test_make_y_binary(self):
        self.assertEqual(self.correct_y_binary, make_y_binary(self.y_class_data_1, self.y_train_1))

    def test_create_predictions(self):
        self.assertEqual(self.predictions, create_predictions(self.x_test, self.clas_pred, self.class_models))

    def test_calc_accuracy(self):
        self.assertEqual(1, calc_accuracy(self.predictions, self.predictions))
        fifty = ["rap", "edm", "rap", "edm", "edm", "rap", "edm", "edm"]
        self.assertEqual(0.5, calc_accuracy(fifty, self.predictions))
        twenty_five = ['rap', 'edm', "edm", 'rap', 'edm', 'rap', 'edm', 'edm']
        self.assertEqual(0.25, calc_accuracy(twenty_five, self.predictions))


if __name__ == '__main__':
    unittest.main()
