from sklearn.datasets import load_digits
from sklearn.model_selection import train_test_split
from evaml.classification import evaluate

# import hand-written digits dataset
dataset = load_digits()
X = dataset.data
y = dataset.target
X_train, X_test, y_train, y_test = train_test_split(X, y, test_size=.2, random_state=42)

print('Evaluating ML Models...')
evaluation_metrics = evaluate(X_train, y_train, X_test, y_test)

for row in evaluation_metrics[0]:
    print(row)
