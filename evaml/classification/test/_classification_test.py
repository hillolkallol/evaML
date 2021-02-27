from evaml.classification import KNearestNeighbors

def _generate_params_test():
    knn = KNearestNeighbors()
    print(knn._generate_params([12,13,14], [1,2,3], [12,13,14], [1,2,3]))


_generate_params_test()