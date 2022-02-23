# SVR

# importing libraries
import numpy as np
import pandas as pd
import matplotlib.pyplot as plt

# importing dataset 
data=pd.read_csv('Position_Salaries.csv')
x=data.iloc[:,1:2].values
y=data.iloc[:,2:3].values


# Feature Scaling
from sklearn.preprocessing import StandardScaler
sc_X = StandardScaler()
sc_y = StandardScaler()
x = sc_X.fit_transform(x)
y = sc_y.fit_transform(y)

# fitting SVR to our dataset
from sklearn.svm import SVR
regressor=SVR(kernel='rbf', degree=5, epsilon=0.1)
regressor.fit(x,y)

# prediction a new result
y_pred=sc_y.inverse_transform(regressor.predict(sc_X.transform([[6.5]])))

# Visualising the SVR results
plt.scatter(x, y, color = 'red')
plt.plot(x, regressor.predict(x), color = 'blue')
plt.title('Truth or Bluff (SVR)')
plt.xlabel('Position level')
plt.ylabel('Salary')
plt.show()

# Visualising the SVR results (for higher resolution and smoother curve)
X_grid = np.arange(min(x), max(x), 0.01) # choice of 0.01 instead of 0.1 step because the data is feature scaled
X_grid = X_grid.reshape((len(X_grid), 1))
plt.scatter(x, y, color = 'red')
plt.plot(X_grid, regressor.predict(X_grid), color = 'blue')
plt.title('Truth or Bluff (SVR)')
plt.xlabel('Position level')
plt.ylabel('Salary')
plt.show()
