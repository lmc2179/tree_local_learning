from sklearn.datasets import load_boston
from sklearn.model_selection import cross_val_score
from tree_local_learning.tree import TreeLocalLearnerRegressor
from sklearn.linear_model import LinearRegression, Ridge, RidgeCV, LassoCV, ElasticNetCV, SGDRegressor
from sklearn.preprocessing import PolynomialFeatures
from sklearn.pipeline import Pipeline
from sklearn.ensemble import RandomForestRegressor, BaggingRegressor
from sklearn.svm import LinearSVR
import numpy as np

X, y = load_boston(return_X_y=True)
#m = LinearRegression()
#m = LinearRegression()
tree_score = cross_val_score(TreeLocalLearnerRegressor(Ridge(), 1./3), X, y, scoring='neg_mean_squared_error', cv=10)
forest_score = cross_val_score(
    BaggingRegressor(
        TreeLocalLearnerRegressor(
            Pipeline([('p', PolynomialFeatures(1)), ('m', RidgeCV())]), 1./3),
              bootstrap_features=True, n_estimators=100), X, y, scoring='neg_mean_squared_error', cv=10)
lr_score = cross_val_score(LinearRegression(), X, y, scoring='neg_mean_squared_error', cv=10)
rf_score = cross_val_score(RandomForestRegressor(n_estimators=100), X, y, scoring='neg_mean_squared_error', cv=10)
print(np.mean(tree_score-lr_score))
print(np.mean(tree_score-rf_score))
print(np.mean(forest_score-lr_score))
print(np.mean(forest_score-rf_score))
print(forest_score)
