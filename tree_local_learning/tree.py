from copy import deepcopy
from sklearn.tree import DecisionTreeRegressor
from collections import defaultdict
import numpy as np
from sklearn.base import BaseEstimator
from sklearn.linear_model import LinearRegression

class TreeLocalLearnerRegressor(BaseEstimator):
    def __init__(self, base_model=None, partition_frac=0.1):
        self.base_model = base_model
        self.partition_frac = partition_frac

    def fit(self, X, y):
        self.tree = DecisionTreeRegressor(min_samples_leaf=int(len(X)*self.partition_frac))
        self.tree.fit(X, y)
        self.partitions = list({idx for idx in self.tree.apply(X)})
        self.partition_models = {}
        row_partition_lookup = defaultdict(list)
        for i, p in enumerate(self.tree.apply(X)):
            row_partition_lookup[p].append(i)
        for p in self.partitions:
            m = deepcopy(self.base_model)
            m.fit(X[row_partition_lookup[p]], y[row_partition_lookup[p]])
            self.partition_models[p] = m
        return self
    
    def predict(self, X):
        predicted_partition = self.tree.apply(X)
        predictions = np.array([self._predict_one([X[i]], p) for i, p in enumerate(predicted_partition)])
        return predictions
        
    def _predict_one(self, row, partition):
        return self.partition_models[partition].predict(np.array(row))[0]
