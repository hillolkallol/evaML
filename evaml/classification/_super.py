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
from sklearn.metrics import accuracy_score

class MLModel:
    """

    :argument
    """
    _START_DATA_SIZE = 25
    _INCREMENT_RATE = 5

    def _generate_evaluation_metrics(self, model, X_train, y_train, X_val, y_val, X_test, y_test):
        """

        :param model:
        :param X_train:
        :param y_train:
        :param X_val:
        :param y_val:
        :return:
        """

        precision, recall, fscore, support, accuracy = self._precision_recall_fscore_support_accuracy(model, X_train, y_train, X_test, y_test)
        learning_curve_data = self._learning_curve_accuracy_measurement(model, X_train, y_train, X_val, y_val)
        return precision, recall, fscore, support, accuracy, learning_curve_data

    def _precision_recall_fscore_support_accuracy(self, model, X_train, y_train, X_test, y_test):
        model.fit(X_train, y_train)
        y_pred = model.predict(X_test)
        precision, recall, fscore, support = precision_recall_fscore_support(y_test, y_pred, average='weighted')
        accuracy = self._calculate_accuracy(model, X_test, y_test)

        return round(precision, 2), round(recall, 2), round(fscore, 2), support, round(accuracy, 2)

    def _learning_curve_accuracy_measurement(self, model, X_train, y_train, X_val, y_val):
        learning_curve_accuracy = []

        start = self._START_DATA_SIZE
        end = len(y_train)
        increment = int((end * self._INCREMENT_RATE) / 100)

        for data_size in range(start, end, increment):
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