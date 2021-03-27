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
import numpy as np
import concurrent.futures
import itertools
from ._super import MLModel


class KNearestNeighbors(MLModel):
    """

    :argument
    """
    def __init__(self,
                 min_neighbors=7,
                 max_neighbors=15, #21,
                 weights=('uniform', 'distance'),
                 algorithms=('auto', 'ball_tree', 'kd_tree', 'brute'),
                 min_leaf_size=20,
                 max_leaf_size=25, #30,
                 min_p=1,
                 max_p=3):
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
        self.__NEIGHBORS = [k_neighbors for k_neighbors in range(min_neighbors, max_neighbors+1, 2)]
        self.__WEIGHTS = weights
        self.__ALGORITHMS = algorithms
        self.__LEAF_SIZE = [leaf_size for leaf_size in range(min_leaf_size, max_leaf_size+1)]
        self.__P = [p for p in range(min_p, max_p+1)]

    def __generate_params(self, X_train, y_train, X_val, y_val, X_test, y_test):
        """

        :param X_train:
        :param y_train:
        :param X_val:
        :param y_val:
        :return:
        """
        return [(X_train, y_train, X_val, y_val, X_test, y_test) + p_tuple
                for p_tuple in
                itertools.product(self.__NEIGHBORS, self.__WEIGHTS, self.__ALGORITHMS, self.__LEAF_SIZE, self.__P)]

    def evaluate_knn(self, param):
        """

        :param param:
        :return:
        """
        X_train, y_train, X_val, y_val, X_test, y_test, k_neighbors, weight, algorithm, leaf_size, p = param

        knn_model = KNeighborsClassifier(n_neighbors=k_neighbors,
                                         weights=weight,
                                         algorithm=algorithm,
                                         leaf_size=leaf_size,
                                         p=p)
        params = {}
        results = {}

        precision, recall, fscore, support, accuracy, learning_curve_data = self._generate_evaluation_metrics(
            knn_model, X_train, y_train, X_val, y_val, X_test, y_test)

        params['neighbors'] = k_neighbors
        params['weight'] = weight
        params['algorithm'] = algorithm
        params['leaf-size'] = leaf_size
        params['p'] = p

        results['precision'] = precision
        results['recall'] = recall
        results['f-score'] = fscore
        results['accuracy'] = accuracy

        learning_curve_plot_name = "learning_curve_" + str(k_neighbors) + "_" \
                                   + str(weight) + "_" + str(algorithm) + "_" + str(leaf_size) + "_" + str(p)

        return params, results, learning_curve_data, learning_curve_plot_name

    def evaluate_knn_multiprocessing(self, X_train, y_train, X_val, y_val, X_test, y_test, reports_per_classifier):
        """

        :param X_train:
        :param y_train:
        :param X_val:
        :param y_val:
        :return:
        """
        params = self.__generate_params(X_train, y_train, X_val, y_val, X_test, y_test)
        evaluation_metrics = {}
        learning_curve_data_all = {}

        with concurrent.futures.ProcessPoolExecutor() as executor:
            output = executor.map(self.evaluate_knn, params)

            param_set = 1
            for parameters, results, learning_curve_data, learning_curve_plot_name in output:

                param_result_set = {}
                param_result_set['params'] = parameters
                param_result_set['results'] = results
                param_result_set['learning_curve_plot_name'] = learning_curve_plot_name

                learning_curve_data_all[learning_curve_plot_name] = learning_curve_data

                evaluation_metrics['param-set-' + str(param_set)] = param_result_set
                param_set += 1

        evaluation_metrics = dict(sorted(evaluation_metrics.items(),
                            reverse=True,
                            key=lambda item: (item[1]['results']['f-score'],
                                              item[1]['results']['accuracy']))[:reports_per_classifier])

        learning_curve_data_minimized = self._learning_curve_data_minimized(learning_curve_data_all,
                                                                            evaluation_metrics)
        return evaluation_metrics, learning_curve_data_minimized

    def _learning_curve_data_minimized(self, learning_curve_data_all, evaluation_metrics):
        learning_curve_data_minimized = {}

        for param_set_key in evaluation_metrics:
            learning_curve_plot_name = evaluation_metrics[param_set_key]['learning_curve_plot_name']
            learning_curve_data_minimized[learning_curve_plot_name] = learning_curve_data_all[learning_curve_plot_name]

        return learning_curve_data_minimized
