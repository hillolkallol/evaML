"""Machine Learning Model and Hyper-parameter Evaluation"""

# Authors: Kallol Das <kalloldash@gmail.com>

from sklearn.model_selection import train_test_split
from evaml.classification import KNearestNeighbors

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
        scores = classifier.evaluate_knn(X, y, X_val, y_val)
        evaluation_metrics.append(scores)

    return evaluation_metrics

def generate_report(evaluation_metrics):
    pass