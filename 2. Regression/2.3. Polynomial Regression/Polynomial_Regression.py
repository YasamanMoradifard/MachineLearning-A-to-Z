'Polynoial linear regression'

# import libraries 
import numpy as np
import pandas as pd 
import matplotlib.pyplot as plt

# import dataset
data=pd.read_csv('Position_Salaries.csv')
x=data.iloc[:,1:2].values
y=data.iloc[:,2:3].values

# visualsing our dataset to find the optimal model of regression
plt.scatter(x,y)
plt.show()

# fitting polynomial linear regression
from sklearn.preprocessing import PolynomialFeatures
reg_poly=PolynomialFeatures(degree=5)
x_poly=reg_poly.fit_transform(x)
from sklearn.linear_model import LinearRegression
reg_li=LinearRegression()
reg_li.fit(x_poly,y)

# visualising the result of polynomial linear regression
plt.scatter(x,y,color='red')
plt.plot(x,reg_li.predict(x_poly),color='blue')
plt.title('polynomial regression result')
plt.xlabel('position')
plt.ylabel('salary')
plt.show()

# Predicting a new result with Linear Regression
level=reg_poly.transform([[6.5]])
lin_reg_predict=reg_li.predict(level)