# Data preprocessing 

## importing libraries
import pandas as pd
import numpy as np
import matplotlib.pyplot as plt

## importing data
Data=pd.read_csv('Data.csv')
X=Data.iloc[:,:-1].values
Y=Data.iloc[:,-1].values

## taking care of MISSING data
from sklearn.impute import SimpleImputer
imputer=SimpleImputer(missing_values=np.nan, strategy="median")
imputer=imputer.fit(X[:,1:3]) # column 1 & 2
X[:,1:3]=imputer.transform(X[:,1:3])

## Encoding categorical data
from sklearn.preprocessing import LabelEncoder, OneHotEncoder
from sklearn.compose import ColumnTransformer

labelencoder_y=LabelEncoder()
Y=labelencoder_y.fit_transform(Y)

labelencoder_x=LabelEncoder()
X[:,0]=labelencoder_x.fit_transform(X[:,0])
ct=ColumnTransformer([("country", OneHotEncoder(),[0])], remainder='passthrough')
X=ct.fit_transform(X)
'''
## feature scaling
from sklearn.preprocessing import StandardScaler
sc=StandardScaler()
X=sc.fit_transform(X)
'''
## splitting dataset into train and test set
from sklearn.model_selection import train_test_split
x_train,x_test,y_train,y_test=train_test_split(X,Y,test_size=0.2)

#Feature Scaling
from sklearn.preprocessing import StandardScaler
sc_X=StandardScaler()
X_train=sc_X.fit_transform(X_train)
X_test=sc_X.transform(X_test)