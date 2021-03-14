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

"""Machine Learning Model and Hyper-parameter Evaluation"""

from sklearn.model_selection import train_test_split
from evaml.classification import KNearestNeighbors
import json
import time
import os

__classifiers_list__ = (KNearestNeighbors(),)

def evaluate(X_train=None,
             y_train=None,
             X_test=None,
             y_test=None,
             classifiers=__classifiers_list__,
             evaluation_size='big',
             directory='evaluation'):
    """

    :param X_train:
    :param y_train:
    :param X_test:
    :param y_test:
    :param classifiers:
    :param evaluation_size:
    :param n_jobs:
    :return:
    """
    evaluation_metrics_all_models = {}
    X, X_val, y, y_val = train_test_split(X_train, y_train, test_size=.2, random_state=42)

    for classifier in __classifiers_list__:

        start = time.time()
        evaluation_metrics = classifier.evaluate_knn_multiprocessing(X, y, X_val, y_val)
        end = time.time()
        print("time taken: ", end - start)

        evaluation_metrics_all_models[classifier.__class__.__name__] = evaluation_metrics

        __create_directories(directory)
        __create_report(evaluation_metrics_all_models, directory)

    return evaluation_metrics_all_models

def __create_report(evaluation_metrics_all_models, directory):
    __create_json_report(evaluation_metrics_all_models, directory)

def __create_json_report(evaluation_metrics_all_models, directory):
    with open(directory + '/report.json', 'w') as f:
        json.dump(evaluation_metrics_all_models, f, indent=4)

    return json.dumps(evaluation_metrics_all_models, indent=4)

def __create_html_report():
    pass

def __create_directories(directory):
    __create_directory(directory)
    __create_directory(directory + '/learning_curves')

def __create_directory(directory):
    if not os.path.exists(directory):
        os.makedirs(directory)

def __generate_summary(evaluation_metrics_all_models):
    pass