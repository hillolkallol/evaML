"""Machine Learning Models"""

# Authors: Kallol Das <kalloldash@gmail.com>

from sklearn.neighbors import KNeighborsClassifier
from sklearn.experimental import enable_halving_search_cv
from sklearn.model_selection import HalvingGridSearchCV
import concurrent.futures
import itertools


class MLModel:
    _START_DATA_SIZE = 25
    _INCREMENT_RATE = 5

    def _grid_search(self, model, param_grid, X, y):
        return HalvingGridSearchCV(model, param_grid).fit(X, y)

    def _learning_curve(self, model, X_train, y_train, X_val, y_val):
        start = self._START_DATA_SIZE
        end = len(y_train)
        increment = int((end * self._INCREMENT_RATE) / 100)
        train_scores = []
        val_scores = []

        for data_size in range(start, end, increment):
            model.fit(X_train[:data_size, :], y_train[:data_size])

            train_score = model.score(X_train[:data_size, :], y_train[:data_size])
            val_score = model.score(X_val, y_val)

            train_scores.append(train_score)
            val_scores.append(val_score)

            # print(data_size, train_score, val_score)

        # pass the scores to visualize the learning curve and store it in the disk
        return val_scores[-1]


class KNearestNeighbors(MLModel):
    """
    parameter : type = default
    --------------------------
    min_neighbors: int = 5,
    max_neighbors: int = 20,
    weights: List[str] = ['uniform', 'distance'],
    algorithms: List[str] = ['auto', 'ball_tree', 'kd_tree', 'brute'],
    min_leaf_size: int = 20,
    max_leaf_size: int = 30,
    min_p: int = 1,
    max_p: int = 5
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
        self.neighbors = [k_neighbors for k_neighbors in range(min_neighbors, max_neighbors+1, 2)]
        self.weights = weights
        self.algorithms = algorithms
        self.leaf_size = [leaf_size for leaf_size in range(min_leaf_size, max_leaf_size+1)]
        self.p = [p for p in range(min_p, max_p+1)]

    """
    
    """
    def evaluate_knn_grid_search(self, X_train, y_train, X_val, y_val):
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

    def evaluate_knn(self, param):
        # print("Evaluating for params: ", k_neighbors, weight, algorithm, leaf_size, p)
        X_train, y_train, X_val, y_val, k_neighbors, weight, algorithm, leaf_size, p = param

        knn_model = KNeighborsClassifier(n_neighbors=k_neighbors,
                                         weights=weight,
                                         algorithm=algorithm,
                                         leaf_size=leaf_size,
                                         p=p)

        knn_model.fit(X_train, y_train)
        score = self._learning_curve(knn_model, X_train, y_train, X_val, y_val) # knn_model.score(X_val, y_val)
        evaluation = [k_neighbors, weight, algorithm, leaf_size, p, str(round(score * 100, 2)) + '%']
        # print(evaluation)
        return evaluation

    def evaluate_knn_multiprocessing(self, X_train, y_train, X_val, y_val):
        params = self._generate_params(X_train, y_train, X_val, y_val)

        with concurrent.futures.ProcessPoolExecutor() as executor:
            results = executor.map(self.evaluate_knn, params)

            # for result in results:
            #     print(result)

    def evaluate_knn_single(self, X_train, y_train, X_val, y_val):
        params = self._generate_params(X_train, y_train, X_val, y_val)

        for param in params:
            self.evaluate_knn(param)

    def _generate_params(self, X_train, y_train, X_val, y_val):
        return [(X_train, y_train, X_val, y_val) + p_tuple
                for p_tuple in
                itertools.product(self.neighbors, self.weights, self.algorithms, self.leaf_size, self.p)]
