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
import concurrent.futures
import itertools
from ._super import MLModel

class KNearestNeighbors(MLModel):
    """

    :argument
    """
    def __init__(self,
                 min_neighbors=7,
                 max_neighbors=10,
                 weights=('uniform', 'distance'),
                 algorithms=('auto', 'ball_tree', 'kd_tree', 'brute'),
                 min_leaf_size=25,
                 max_leaf_size=27,
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

    def __generate_params(self, X_train, y_train, X_val, y_val):
        """

        :param X_train:
        :param y_train:
        :param X_val:
        :param y_val:
        :return:
        """
        return [(X_train, y_train, X_val, y_val) + p_tuple
                for p_tuple in
                itertools.product(self.__NEIGHBORS, self.__WEIGHTS, self.__ALGORITHMS, self.__LEAF_SIZE, self.__P)]

    def __evaluate_knn(self, param):
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
        params = self.__generate_params(X_train, y_train, X_val, y_val)
        analysis_result = {}

        with concurrent.futures.ProcessPoolExecutor() as executor:
            results = executor.map(self.__evaluate_knn, params)

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
        params = self.__generate_params(X_train, y_train, X_val, y_val)
        analysis_result = {}

        with concurrent.futures.ThreadPoolExecutor() as executor:
            results = executor.map(self.__evaluate_knn, params)

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
        params = self.__generate_params(X_train, y_train, X_val, y_val)
        analysis_result = {}

        param_set = 1
        for param in params:
            result = self.__evaluate_knn(param)
            analysis_result['param-set-' + str(param_set)] = result
            param_set += 1

        return analysis_result
