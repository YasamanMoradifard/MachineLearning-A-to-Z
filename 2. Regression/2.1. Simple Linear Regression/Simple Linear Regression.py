#Simple Linear Regression

# importing libraries
import numpy as np 
import pandas as pd
import matplotlib.pyplot as plt

# import dataset
dataset=pd.read_csv('Salary_Data.csv')
X=dataset.iloc[:,0].values
Y=dataset.iloc[:,1].values

# Preprocessing
## feature scaling 
from sklearn.preprocessing import  StandardScaler
SC_X=StandardScaler()
X=SC_X.fit_transform(X.reshape(-1,1))

SC_Y=StandardScaler()
Y=SC_Y.fit_transform(Y.reshape(-1,1))

## splitting dataset into test and train sets
from sklearn.model_selection import train_test_split
X_train,X_test,Y_train,Y_test=train_test_split(X,Y,test_size=1/6) 

# fiting Linear Regression to train set
from sklearn.linear_model import LinearRegression
regressor=LinearRegression()
regressor.fit(X_train.reshape(-1,1),Y_train.reshape(-1,1))

# Predicting the test set results
Y_pred=regressor.predict(X_test.reshape(-1,1))

# Calculating loss
loss=sum(Y_pred**2-Y_test**2)

#Visualizing the results from training set
plt.scatter(X_train,Y_train,color='red')
plt.plot(X_train,regressor.predict(X_train),color='blue')
plt.title('Salary vs Experience (Training set)')
plt.xlabel('Years of Experience')
plt.ylabel('Salary')
plt.show()

#Visualizing the results from test set
plt.scatter(X_test,Y_test,color='red')
plt.plot(X_train,regressor.predict(X_train),color='blue')
plt.title('Salary vs Experience (test set)')
plt.xlabel('Years of Experience')
plt.ylabel('Salary')
plt.show()
