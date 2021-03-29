# ===============================================================================
# MIT License
#
# Copyright (c) 2021 Kallol Das
#
# Permission is hereby granted, free of charge, to any person obtaining a copy
# of this software and associated documentation files (the "Software"), to deal
# in the Software without restriction, including without limitation the rights
# to use, copy, modify, merge, publish, distribute, sublicense, and/or sell
# copies of the Software, and to permit persons to whom the Software is
# furnished to do so, subject to the following conditions:
#
# The above copyright notice and this permission notice shall be included in all
# copies or substantial portions of the Software.
#
# THE SOFTWARE IS PROVIDED "AS IS", WITHOUT WARRANTY OF ANY KIND, EXPRESS OR
# IMPLIED, INCLUDING BUT NOT LIMITED TO THE WARRANTIES OF MERCHANTABILITY,
# FITNESS FOR A PARTICULAR PURPOSE AND NONINFRINGEMENT. IN NO EVENT SHALL THE
# AUTHORS OR COPYRIGHT HOLDERS BE LIABLE FOR ANY CLAIM, DAMAGES OR OTHER
# LIABILITY, WHETHER IN AN ACTION OF CONTRACT, TORT OR OTHERWISE, ARISING FROM,
# OUT OF OR IN CONNECTION WITH THE SOFTWARE OR THE USE OR OTHER DEALINGS IN THE
# SOFTWARE.
# ===============================================================================
# Authors: Kallol Das <kalloldash@gmail.com>
# ===============================================================================

"""Machine Learning Model Super Class"""

from sklearn.metrics import classification_report
from sklearn.metrics import confusion_matrix
from sklearn.metrics import precision_recall_fscore_support


class MLModel:
    def __init__(self, learning_curve_min_data_size, learning_curve_increment_rate):
        """
        Initializes the ML Model super class instances.

        :param learning_curve_min_data_size:
            Data Type - Integer.
            The minimum data size for the learning curve.
            If not given, the default value will be used. Default is 25.

        :param learning_curve_increment_rate:
            Data Type - Integer.
            The increment rate for the learning curve.
            If not given, the default value will be used. Default is 25.
        """
        self.learning_curve_min_data_size = learning_curve_min_data_size
        self.learning_curve_increment_rate = learning_curve_increment_rate

    def _generate_evaluation_metrics(self, model, X_train, y_train, X_val, y_val, X_test, y_test):
        """
        Generates evaluation metrics (precision, recall, f-score, support, accuracy,
        and generates learning curve data.)

        :param model:
            Data Type - Object like.
            Machine learning classifier model.

        :param X_train:
            Data Type - Array like.
            X coordinates of training dataset.

        :param y_train:
            Data Type - Array like.
            y coordinates of training dataset.

        :param X_val:
            Data Type - Array like.
            X coordinates of validation dataset.

        :param y_val:
            Data Type - Array like.
            y coordinates of validation dataset.

        :param X_test:
            Data Type - Array like.
            X coordinates of test dataset.

        :param y_test:
            Data Type - Array like.
            y coordinates of test dataset.

        :return:
            returns precision, recall, f-score, support, accuracy,
            and generates learning curve data (training and validation accuracy).
        """
        precision, recall, fscore, support, accuracy = self._precision_recall_fscore_support_accuracy(model,
                                                        X_train, y_train, X_test, y_test)
        learning_curve_data = self._learning_curve_accuracy_measurement(model, X_train, y_train, X_val, y_val)
        return precision, recall, fscore, support, accuracy, learning_curve_data

    def _learning_curve_data_minimized(self, learning_curve_data_all, evaluation_metrics):
        """
        Minimizes the learning curve data and returns the learning curve data for top k models.

        :param learning_curve_data_all:
            Data Type - List of tuples.
            list of tuples that contains training and validation accuracy.

        :param evaluation_metrics:
            Data Type - Dictionary.
            dictionary that contains the evaluation result of the ML model for different set of hyper-parameters.

        :return:
            Top k-tuples that contains training and validation accuracy.
        """
        learning_curve_data_minimized = dict()

        for param_set_key in evaluation_metrics:
            learning_curve_plot_name = evaluation_metrics[param_set_key]['learning_curve_plot_name']
            learning_curve_data_minimized[learning_curve_plot_name] = learning_curve_data_all[learning_curve_plot_name]

        return learning_curve_data_minimized

    def _precision_recall_fscore_support_accuracy(self, model, X_train, y_train, X_test, y_test):
        """
        Calculates and returns precision, recall, f-score, support, accuracy

        :param model:
            Data Type - Object like.
            Machine learning classifier model.

        :param X_train:
            Data Type - Array like.
            X coordinates of training dataset.

        :param y_train:
            Data Type - Array like.
            y coordinates of training dataset.

        :param X_test:
            Data Type - Array like.
            X coordinates of test dataset.

        :param y_test:
            Data Type - Array like.
            y coordinates of test dataset.

        :return:
            returns precision, recall, f-score, support, accuracy
        """
        model.fit(X_train, y_train)
        y_pred = model.predict(X_test)
        precision, recall, fscore, support = precision_recall_fscore_support(y_test, y_pred, average='weighted')
        accuracy = self._calculate_accuracy(model, X_test, y_test)

        return round(precision, 2), round(recall, 2), round(fscore, 2), support, round(accuracy, 2)

    def _learning_curve_accuracy_measurement(self, model, X_train, y_train, X_val, y_val):
        """
        Measures and returns list of tuples that contains training and validation accuracy.

        :param model:
            Data Type - Object like.
            Machine learning classifier model.

        :param X_train:
            Data Type - Array like.
            X coordinates of training dataset.

        :param y_train:
            Data Type - Array like.
            y coordinates of training dataset.

        :param X_val:
            Data Type - Array like.
            X coordinates of validation dataset.

        :param y_val:
            Data Type - Array like.
            y coordinates of validation dataset.

        :return:
            Returns list of tuples that contains training and validation accuracy.
        """
        learning_curve_accuracy = []

        start = self.learning_curve_min_data_size
        end = len(y_train)
        increment = int((end * self.learning_curve_increment_rate) / 100)

        for idx in range(start, end + increment, increment): # [10, 15, 20, 25, 30, 35], 10 --- 45
            data_size = min(idx, end)

            model.fit(X_train[:data_size, :], y_train[:data_size])
            train_accuracy = self._calculate_accuracy(model, X_train[:data_size, :], y_train[:data_size])
            val_accuracy = self._calculate_accuracy(model, X_val, y_val)

            learning_curve_accuracy.append([data_size, round(train_accuracy, 2), round(val_accuracy, 2)])

        return learning_curve_accuracy

    def _calculate_accuracy(self, model, X, y):
        return model.score(X, y)

    def _classification_report(self, y_true, y_pred):
        return classification_report(y_true, y_pred, output_dict=True)

    def _generate_confusion_matrix(self, y_true, y_pred):
        return confusion_matrix(y_true, y_pred)

    def set_learning_curve_min_data_size(self, size):
        self.learning_curve_min_data_size = size

    def set_learning_curve_increment_rate(self, rate):
        self.learning_curve_increment_rate = rate