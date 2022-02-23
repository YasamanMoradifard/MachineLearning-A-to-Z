# Decision Tree Regression

# importing libraries
import numpy as np
import matplotlib.pyplot as plt
import pandas as pd

# import the dataset
data=pd.read_csv('Position_Salaries.csv')
x=data.iloc[:,1:2].values
y=data.iloc[:,2:3].values

'''
# Feature Scaling
from sklearn.preprocessing import StandardScaler
sc_X = StandardScaler()
x= sc_X.fit_transform(x)
sc_y = StandardScaler()
y= sc_y.fit_transform(y)
'''
# fitting decision tree 
from sklearn.tree import DecisionTreeRegressor
regressor=DecisionTreeRegressor()
regressor.fit(x, y)

# Predicting a new result
y_pred =regressor.predict([[6.5]])

# Visualising the Decision Tree Regression results (higher resolution)
X_grid = np.arange(min(x), max(x), 0.001)
X_grid = X_grid.reshape((len(X_grid), 1))
plt.scatter(x, y, color = 'red')
plt.plot(X_grid, regressor.predict(X_grid), color = 'blue')
plt.title('Truth or Bluff (Decision Tree Regression)')
plt.xlabel('Position level')
plt.ylabel('Salary')
plt.show()