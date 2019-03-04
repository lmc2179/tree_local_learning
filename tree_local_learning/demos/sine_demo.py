from tree_local_learning.tree import TreeLocalLearnerRegressor

from sklearn.linear_model import LinearRegression
from sklearn.preprocessing import PolynomialFeatures
from sklearn.pipeline import Pipeline
from matplotlib import pyplot as plt 
import seaborn as sns
import numpy as np
from sklearn.model_selection import cross_val_predict

X = np.array(sorted(np.random.normal(0,1, 100)))
X_m = X.reshape(-1, 1)
y = np.sin(X)
p = Pipeline([('p', PolynomialFeatures()),
              ('m', LinearRegression())])
m = TreeLocalLearnerRegressor(p, 0.25)
m.fit(X_m, y)
plt.scatter(X, y)
#y_pred = m.predict(X_m)
y_pred = cross_val_predict(m, X_m, y, cv=50)
plt.plot(X, y_pred, color='orange', label='Tree of local regression')
y_pred = cross_val_predict(p, X_m, y, cv=50)
plt.plot(X, y_pred, color='green', label='Quadratic Regression')
plt.show()
