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
import matplotlib.pyplot as plt
import numpy as np
import logging
import json
import time
import os

####################################
# Will separate the logger later
# create logger
logger = logging.getLogger(__name__)
logger.setLevel(logging.INFO)

# create console handler and set level to debug
ch = logging.StreamHandler()
ch.setLevel(logging.INFO)

# create formatter
formatter = logging.Formatter('%(asctime)s - %(name)s - %(levelname)s - %(message)s')

# add formatter to ch
ch.setFormatter(formatter)

# add ch to logger
logger.addHandler(ch)
###################################

__classifiers_list__ = (KNearestNeighbors(),)


def evaluate(X_train=None,
             y_train=None,
             X_test=None,
             y_test=None,
             classifiers=__classifiers_list__,
             directory='evaluation'):
    """

    :param X_train:
    :param y_train:
    :param X_test:
    :param y_test:
    :param classifiers:
    :param directory:
    :return:
    """
    __create_directories(directory)
    evaluation_metrics_all_models = {}
    X, X_val, y, y_val = train_test_split(X_train, y_train, test_size=.2, random_state=42)

    for classifier in classifiers:

        start = time.time()
        evaluation_metrics, learning_curve_data_all = classifier.evaluate_knn_multiprocessing(X, y, X_val, y_val, X_test, y_test)
        end = time.time()
        logger.info(classifier.__class__.__name__ + " >>> Time taken: " + str(round(end - start, 2)))

        __plot_learning_curves(learning_curve_data_all, directory)
        evaluation_metrics_all_models[classifier.__class__.__name__] = evaluation_metrics

    return __create_report(evaluation_metrics_all_models, directory)


def __plot_learning_curves(learning_curve_data_all, directory):
    for learning_curve_data_name in learning_curve_data_all:
        # print(learning_curve_data_name)
        __plot_learning_curve(np.array(learning_curve_data_all[learning_curve_data_name]), learning_curve_data_name, directory)


def __plot_learning_curve(learning_curve_data, learning_curve_data_name, directory):

    datasize = learning_curve_data[:, 0]
    train_accuracy = learning_curve_data[:, 1]
    val_accuracy = learning_curve_data[:, 2]

    fig, ax = plt.subplots()

    line1, = ax.plot(datasize, train_accuracy, label='Training Set Accuracy')
    line2, = ax.plot(datasize, val_accuracy, label='Validation Set Accuracy')

    ax.legend()
    plt.savefig(directory + '/learning_curves/' + learning_curve_data_name)


def __create_report(evaluation_metrics_all_models, directory):
    __create_json_report(evaluation_metrics_all_models, directory)
    __create_html_report(directory)


def __create_json_report(evaluation_metrics_all_models, directory):
    with open(directory + '/report.json', 'w') as f:
        json.dump(evaluation_metrics_all_models, f, indent=4)

    return json.dumps(evaluation_metrics_all_models, indent=4)


def __read_json_report(directory):
    with open(directory + '/report.json', 'r') as f:
        json_report = json.load(f)

    return json_report


def __create_html_report(directory):

    """

    :param directory:
    :return:
    """
    json_report = __read_json_report(directory)

    html_report = """
    
    """

    with open(directory + '/report.html', 'w') as file:
        file.write(html_report)


def __create_directories(directory):
    __create_directory(directory)
    __create_directory(directory + '/learning_curves')


def __create_directory(directory):
    if not os.path.exists(directory):
        os.makedirs(directory)


def __generate_summary():
    pass