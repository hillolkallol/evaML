## evaML

evaML is an Open Source Python Machine Learning Framework to evaluate and compare Machine Learning Models and Hyperparameters. All is needed to call the evaluate() function and it will run Machine Learning Models with different set of hyperparameters parallelly; compare and generate a report (full or summary).

### Motivation
Industry and University research spend a lot of time on evaluating, comparing and finding best ML Models and Hyperparameters. This framework does it automatically just by calling a single function.

### How to use evaML?
evaML is still not available in Python Package Index - pypi, but it still can be used by putting the repository folder in python site-package.

#### Example
Please note that it is mendatory in windows to keep evaluate() function inside ```__name__ == '__main__':``` and also recommended in Linux.

```python
from sklearn import datasets
from sklearn.model_selection import train_test_split
from evaml.classification import evaluate


# import dataset
iris = datasets.load_iris()
X = iris.data[:, :2]  # we only take the first two features.
y = iris.target
X_train, X_test, y_train, y_test = train_test_split(X, y, test_size=.2, random_state=42)

if __name__ == '__main__':
    evaluation_metrics_all_models = evaluate(X_train, y_train, X_test, y_test)
    # print(evaluation_metrics_all_models)
```

### Evaluation Metrics
Classification: Accuracy, Precision, Recall, F-Score, Confusion Matrix, Learning Curve

### Contribution Guidelines
Will add guideline soon so that anyone can fork and contribute.
