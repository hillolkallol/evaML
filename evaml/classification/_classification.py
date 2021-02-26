"""Machine Learning Models"""

# Authors: Kallol Das <kalloldash@gmail.com>

from sklearn.neighbors import KNeighborsClassifier

class KNearestNeighbors:
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
    def __init__(self, min_neighbors=5,
                 max_neighbors=20,
                 weights=['uniform', 'distance'],
                 algorithms=['auto', 'ball_tree', 'kd_tree', 'brute'],
                 min_leaf_size=20,
                 max_leaf_size=30,
                 min_p=1,
                 max_p=5):
        self.min_neighbors = min_neighbors
        self.max_neighbors = max_neighbors
        self.weights = weights
        self.algorithms = algorithms
        self.min_leaf_size = min_leaf_size
        self.max_leaf_size = max_leaf_size
        self.min_p = min_p
        self.max_p = max_p

    """
    
    """
    def evaluate_knn(self, X_train, y_train, X_val, y_val):
        scores = [['k_neighbors', 'weight', 'algorithm', 'leaf_size', 'p', 'score'],
                  ['-----------', '------', '---------', '---------', '-', '-----']]
        for k_neighbors in range(self.min_neighbors, self.max_neighbors):
            for weight in self.weights:
                for algorithm in self.algorithms:
                    for leaf_size in range(self.min_leaf_size, self.max_leaf_size):
                        for p in range(self.min_p, self.max_p):
                            knn_model = KNeighborsClassifier(n_neighbors=k_neighbors,
                                                             weights=weight,
                                                             algorithm=algorithm,
                                                             leaf_size=leaf_size,
                                                             p=p)
                            knn_model.fit(X_train, y_train)
                            score = knn_model.score(X_val, y_val)
                            evaluation = [k_neighbors, weight, algorithm, leaf_size, p, str(round(score*100, 2)) + '%']
                            scores.append(evaluation)
        return scores

    def learning_curve(self, ):
