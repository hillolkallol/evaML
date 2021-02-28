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

"""Machine Learning Models"""

from sklearn.neighbors import KNeighborsClassifier
from sklearn.experimental import enable_halving_search_cv
from sklearn.model_selection import HalvingGridSearchCV
import concurrent.futures
import itertools
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
            confusion_matrix = self._generate_confusion_matrix(y_val, y_pred)

            analysis = {'classification-report' : classification_report,
                        'confusion-matrix' : confusion_matrix}

            metrics_analysis['data-size-' + str(data_size)] = analysis

        return metrics_analysis

    def _calculate_accuracy(self, model, X, y, data_size):
        return model.score(X[:data_size, :], y[:data_size])

    def _classification_report(self, y_true, y_pred):
        return classification_report(y_true, y_pred)

    def _generate_confusion_matrix(self, y_true, y_pred):
        return confusion_matrix(y_true, y_pred)


class KNearestNeighbors(MLModel):
    """

    :argument
    """
    def __init__(self,
                 min_neighbors=5,
                 max_neighbors=21,
                 weights=['uniform', 'distance'],
                 algorithms=['auto', 'ball_tree', 'kd_tree', 'brute'],
                 min_leaf_size=20,
                 max_leaf_size=30,
                 min_p=1,
                 max_p=5):
        """

        :param min_neighbors:
        :param max_neighbors:
        :param weights:
        :param algorithms:
        :param min_leaf_size:
        :param max_leaf_size:
        :param min_p:
        :param max_p:
        """
        self.neighbors = [k_neighbors for k_neighbors in range(min_neighbors, max_neighbors+1, 2)]
        self.weights = weights
        self.algorithms = algorithms
        self.leaf_size = [leaf_size for leaf_size in range(min_leaf_size, max_leaf_size+1)]
        self.p = [p for p in range(min_p, max_p+1)]

    def evaluate_knn(self, param):
        """

        :param param:
        :return:
        """
        X_train, y_train, X_val, y_val, k_neighbors, weight, algorithm, leaf_size, p = param

        knn_model = KNeighborsClassifier(n_neighbors=k_neighbors,
                                         weights=weight,
                                         algorithm=algorithm,
                                         leaf_size=leaf_size,
                                         p=p)
        evaluation_metrics = {}

        metrics_analysis = self._generate_evaluation_metrics(knn_model, X_train, y_train, X_val, y_val)

        evaluation_metrics['neighbors'] = k_neighbors
        evaluation_metrics['weight'] = weight
        evaluation_metrics['algorithm'] = algorithm
        evaluation_metrics['leaf-size'] = leaf_size
        evaluation_metrics['p'] = p
        evaluation_metrics['metrics_analysis'] = metrics_analysis

        return evaluation_metrics

    def evaluate_knn_multiprocessing(self, X_train, y_train, X_val, y_val):
        """

        :param X_train:
        :param y_train:
        :param X_val:
        :param y_val:
        :return:
        """
        params = self._generate_params(X_train, y_train, X_val, y_val)
        analysis_result = {}

        with concurrent.futures.ProcessPoolExecutor() as executor:
            results = executor.map(self.evaluate_knn, params)

            param_set = 1
            for result in results:
                analysis_result['param-set-' + str(param_set)] = result
                param_set += 1

        return analysis_result

    def evaluate_knn_multithreading(self, X_train, y_train, X_val, y_val):
        """

        :param X_train:
        :param y_train:
        :param X_val:
        :param y_val:
        :return:
        """
        params = self._generate_params(X_train, y_train, X_val, y_val)
        analysis_result = {}

        with concurrent.futures.ThreadPoolExecutor() as executor:
            results = executor.map(self.evaluate_knn, params)

            param_set = 1
            for result in results:
                analysis_result['param-set-' + str(param_set)] = result
                param_set += 1

        return analysis_result

    def evaluate_knn_singleprocessing(self, X_train, y_train, X_val, y_val):
        """

        :param X_train:
        :param y_train:
        :param X_val:
        :param y_val:
        :return:
        """
        params = self._generate_params(X_train, y_train, X_val, y_val)
        analysis_result = {}

        param_set = 1
        for param in params:
            result = self.evaluate_knn(param)
            analysis_result['param-set-' + str(param_set)] = result
            param_set += 1

        return analysis_result

    def _generate_params(self, X_train, y_train, X_val, y_val):
        """

        :param X_train:
        :param y_train:
        :param X_val:
        :param y_val:
        :return:
        """
        return [(X_train, y_train, X_val, y_val) + p_tuple
                for p_tuple in
                itertools.product(self.neighbors, self.weights, self.algorithms, self.leaf_size, self.p)]

    def evaluate_knn_grid_search(self, X_train, y_train, X_val, y_val):
        """
        Not using grid search at this point. It might be useful though in the future. So keeping it for now.
        :param X_train:
        :param y_train:
        :param X_val:
        :param y_val:
        :return:
        """
        knn_model = KNeighborsClassifier()
        param_grid = {'n_neighbors' : self.neighbors,
                     'weights' : self.weights,
                     'algorithm' : self.algorithms,
                     'leaf_size' : self.leaf_size,
                     'p' : self.p}

        halving_grid_search = self._grid_search(knn_model, param_grid, X_train, y_train)
        best_model = halving_grid_search.best_estimator_
        score = halving_grid_search.best_score_
        print(halving_grid_search.cv_results_)
