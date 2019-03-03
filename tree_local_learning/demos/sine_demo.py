from tree_local_learning.tree import TreeLocalRegressor

from sklearn.linear_model import LinearRegression
from sklearn.preprocessing import PolynomialFeatures
from sklearn.pipeline import Pipeline
from matplotlib import pyplot as plt 
import seaborn as sns

X = np.random.normal(0,1, 100)
X_m = X.reshape(-1, 1)
y = np.sin(X)
p = Pipeline([('p', PolynomialFeatures()),
              ('m', LinearRegression())])
m = TreeLocalLearnerRegressor(p, 0.25)
m.fit(X_m, y)
plt.scatter(X, y)
plt.scatter(X, m.predict(X_m), color='orange')
plt.show()
