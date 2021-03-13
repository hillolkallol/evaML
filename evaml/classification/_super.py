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

from sklearn.experimental import enable_halving_search_cv
from sklearn.model_selection import HalvingGridSearchCV
from sklearn.metrics import classification_report
from sklearn.metrics import confusion_matrix

class MLModel:
    """

    :argument
    """
    _START_DATA_SIZE = 25
    _INCREMENT_RATE = 5

    def _grid_search(self, model, param_grid, X, y):
        """

        :param model:
        :param param_grid:
        :param X:
        :param y:
        :return:
        """

        return HalvingGridSearchCV(model, param_grid).fit(X, y)

    def _generate_evaluation_metrics(self, model, X_train, y_train, X_val, y_val):
        """

        :param model:
        :param X_train:
        :param y_train:
        :param X_val:
        :param y_val:
        :return:
        """
        metrics_analysis = {}

        start = self._START_DATA_SIZE
        end = len(y_train)
        increment = int((end * self._INCREMENT_RATE) / 100)

        for data_size in range(start, end, increment):
            model.fit(X_train[:data_size, :], y_train[:data_size])
            y_pred = model.predict(X_val)

            classification_report = self._classification_report(y_val, y_pred)
            # confusion_matrix = self._generate_confusion_matrix(y_val, y_pred)

            analysis = {'classification-report' : classification_report}
                        # 'confusion-matrix' : confusion_matrix}

            metrics_analysis['data-size-' + str(data_size)] = analysis

        return metrics_analysis

    def _calculate_accuracy(self, model, X, y, data_size):
        return model.score(X[:data_size, :], y[:data_size])

    def _classification_report(self, y_true, y_pred):
        return classification_report(y_true, y_pred, output_dict=True)

    def _generate_confusion_matrix(self, y_true, y_pred):
        return confusion_matrix(y_true, y_pred)