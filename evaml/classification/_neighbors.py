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
    def __init__(self,
                 min_neighbors=7,
                 max_neighbors=21,
                 weights=('uniform', 'distance'),
                 algorithms=('auto', 'ball_tree', 'kd_tree', 'brute'),
                 min_leaf_size=20,
                 max_leaf_size=30,
                 min_p=1,
                 max_p=5,
                 learning_curve_min_data_size=25,
                 learning_curve_increment_rate=20):
        """
        Initializes the K-Nearest Neighbors instances.
        Default hyper-parameters of K-Nearest Neighbors are set through constructor.
        Gives a flexibility of choosing wide range of hyper-parameters to tune.

        :param min_neighbors:
            Data Type - Integer.
            The minimum number of neighbors that is used in sklearn k-nearest neighbors classifier.
            If not given, the default value will be used. Default is 7.

        :param max_neighbors:
            Data Type - Integer.
            The maximum number of neighbors that is used in sklearn k-nearest neighbors classifier.
            If not given, the default value will be used. Default is 21.

        :param weights:
            Data Type - Tuple.
            The tuple of weights that are used in sklearn k-nearest neighbors classifier.
            If not given, the default values will be used. Default is ('uniform', 'distance').

        :param algorithms:
            Data Type - Tuple.
            The tuple of algorithms that are used in sklearn k-nearest neighbors classifier.
            If not given, the default values will be used. Default is ('auto', 'ball_tree', 'kd_tree', 'brute')

        :param min_leaf_size:
            Data Type - Integer.
            The minimum number of leaf size that is used in sklearn k-nearest neighbors classifier.
            If not given, the default value will be used. Default is 20.

        :param max_leaf_size:
            Data Type - Integer.
            The maximum number of leaf size that is used in sklearn k-nearest neighbors classifier.
            If not given, the default value will be used. Default is 30.

        :param min_p:
            Data Type - Integer.
            The minimum number of power parameter for the Minkowski metric
            that is used in sklearn k-nearest neighbors classifier.
            If not given, the default value will be used. Default is 1.

        :param max_p:
            Data Type - Integer.
            The minimum number of power parameter for the Minkowski metric
            that is used in sklearn k-nearest neighbors classifier.
            If not given, the default value will be used. Default is 5.

        :param learning_curve_min_data_size:
            Data Type - Integer.
            The minimum data size for the learning curve.
            If not given, the default value will be used. Default is 25.

        :param learning_curve_increment_rate:
            Data Type - Integer.
            The increment rate for the learning curve.
            If not given, the default value will be used. Default is 25.
        """

        super().__init__(learning_curve_min_data_size, learning_curve_increment_rate)

        self.__NEIGHBORS = [k_neighbors for k_neighbors in range(min_neighbors, max_neighbors+1, 2)]
        self.__WEIGHTS = weights
        self.__ALGORITHMS = algorithms
        self.__LEAF_SIZE = [leaf_size for leaf_size in range(min_leaf_size, max_leaf_size+1)]
        self.__P = [p for p in range(min_p, max_p+1)]

    def evaluate_model_multiprocessing(self, X_train, y_train, X_val, y_val, X_test, y_test, reports_per_classifier):
        """
        Evaluates models for different set of hyper-parameters parallelly.

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

        :param reports_per_classifier:
            Data Type - Integer.
            Number of top results per classifier that are picked to add in the report.

        :return:
            Returns evaluation result- evaluation_metrics, learning_curve_data_minimized.
            evaluation_metrics - dictionary that contains the evaluation result of the ML model
            for different set of hyper-parameters.
            learning_curve_data_minimized - top n tuples that contains training and validation accuracy.
        """

        params = self.__generate_params(X_train, y_train, X_val, y_val, X_test, y_test)
        evaluation_metrics = dict()
        learning_curve_data_all = dict()

        with concurrent.futures.ProcessPoolExecutor() as executor:
            output = executor.map(self.evaluate_model, params)

            param_set = 1
            for parameters, results, learning_curve_data, learning_curve_plot_name in output:

                param_result_set = dict()
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

    def evaluate_model(self, param):
        """

        :param param:
            Data Type - Tuple.
            param tuple contains-
            X_train, y_train, X_val, y_val, X_test, y_test, k_neighbors, weight, algorithm, leaf_size, p.
            X_train - X coordinates of training dataset.
            y_train - y coordinates of training dataset.
            X_val - X coordinates of validation dataset.
            y_val - y coordinates of validation dataset.
            X_test - X coordinates of test dataset.
            y_test - y coordinates of test dataset.
            k_neighbors - number of neighbors for k-nearest neighbors classifier.
            weight - weight that is used in sklearn k-nearest neighbors classifier.
            algorithm - algorithm that are used in sklearn k-nearest neighbors classifier.
            leaf_size - leaf size that is used in sklearn k-nearest neighbors classifier.
            p - number of power parameter for the Minkowski metric.

        :return:
            Returns evaluation result- params, results, learning_curve_data, learning_curve_plot_name.
            params - dictionary that contains the hyper-parameters that are used.
            results - dictionary that contains the evaluation result of the ML model.
            learning_curve_data - list of tuples that contains training and validation accuracy.
            learning_curve_plot_name - learning curve plot jpg name.
        """

        X_train, y_train, X_val, y_val, X_test, y_test, k_neighbors, weight, algorithm, leaf_size, p = param

        knn_model = KNeighborsClassifier(n_neighbors=k_neighbors,
                                         weights=weight,
                                         algorithm=algorithm,
                                         leaf_size=leaf_size,
                                         p=p)
        params = dict()
        results = dict()

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

        learning_curve_plot_name = "learning_curve_" + str(knn_model.__class__.__name__) + "_" + str(k_neighbors) \
                                   + "_" + str(weight) + "_" + str(algorithm) + "_" + str(leaf_size) + "_" + str(p)

        return params, results, learning_curve_data, learning_curve_plot_name

    def __generate_params(self, X_train, y_train, X_val, y_val, X_test, y_test):
        """
        Generates combinations (tuples) of hyper-parameters

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
            List of tuples.
            Combinations (tuples) of hyper-parameters.
        """

        return [(X_train, y_train, X_val, y_val, X_test, y_test) + p_tuple
                for p_tuple in
                itertools.product(self.__NEIGHBORS, self.__WEIGHTS, self.__ALGORITHMS, self.__LEAF_SIZE, self.__P)]