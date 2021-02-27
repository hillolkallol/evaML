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
import time

__classifiers_list__ = [KNearestNeighbors()]

"""
PARAMETERS:
----------
X_train : array like, default=None
    
y_train : array like, default=None

X_test : array like, default=None

y_test : array like, default=None

classifiers : list, default=classifiers_list
    List of all the ML classifiers

evaluation_size : string, default='big'
    Iterations vary based on this parameter. Values are: small, medium, big.
    
n_jobs : int, default=10
    Runs at most n_jobs parallelly
"""
def evaluate(X_train=None,
             y_train=None,
             X_test=None,
             y_test=None,
             classifiers=__classifiers_list__,
             evaluation_size='big',
             n_jobs=10):

    evaluation_metrics = []
    X, X_val, y, y_val = train_test_split(X_train, y_train, test_size=.2, random_state=42)

    for classifier in __classifiers_list__:
        start = time.time()
        scores = classifier.evaluate_knn_multiprocessing(X, y, X_val, y_val)
        end = time.time()
        print("MP time taken: ", end - start)

        evaluation_metrics.append(scores)

    return evaluation_metrics

def generate_report(evaluation_metrics):
    pass