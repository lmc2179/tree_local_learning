from tree_local_learning.tree import TreeLocalLearnerRegressor

from sklearn.linear_model import LinearRegression
from sklearn.preprocessing import PolynomialFeatures
from sklearn.pipeline import Pipeline
from matplotlib import pyplot as plt 
import seaborn as sns
import numpy as np
from sklearn.model_selection import cross_val_predict

X = np.array(sorted(np.random.normal(0,2, 250)))
X_m = X.reshape(-1, 1)
y = np.sin(X)
p = Pipeline([('p', PolynomialFeatures()),
              ('m', LinearRegression())])
m = TreeLocalLearnerRegressor(LinearRegression(), 0.01)
m.fit(X_m, y)
plt.scatter(X, y)
#y_pred = m.predict(X_m)
y_pred = cross_val_predict(m, X_m, y, cv=250)
plt.plot(X, y_pred, color='orange', label='Tree of local linear regression')
y_pred = cross_val_predict(LinearRegression(), X_m, y, cv=250)
plt.plot(X, y_pred, color='green', label='Linear Regression')
plt.legend()
plt.title('Leave one out cross validation')
plt.show()
