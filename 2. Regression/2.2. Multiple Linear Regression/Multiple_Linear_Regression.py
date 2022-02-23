'Multiple linear regression'
# NUMBER 1
'''
# importing librries
import numpy as np
import matplotlib.pyplot as plt
import pandas as pd

### importing dataset
data=pd.read_csv('50_Startups.csv')
x=data.iloc[:,:-1].values
y=data.iloc[:,-1].values

### preprocessing
# missing data
from sklearn.preprocessing import SimpleImputer
imputer_x=SimpleImputer()
X=imputer_x.fit_transform(x)

imputer_y=SimpleImputer()
y=imputer_y.fit_transform(y)

# Encoding categorical data
from sklearn.preprocessing import LabelEncoder, OneHotEncoder
from sklearn.compose import ColumnTransformer

lableencoder=LabelEncoder()
x[:,3]=lableencoder.fit_transform(x[:,3])

CT=ColumnTransformer([("Country",OneHotEncoder(),[3])],remainder='passthrough')
x=CT.fit_transform(x)

# Avoiding the Dummy variables Trap
x=x[:,1:]

# Splitting dataset into Train and Test sets
from sklearn.model_selection import train_test_split
x_tr,x_ts,y_tr,y_ts=train_test_split(x,y,test_size=0.2)

# feature scaling
from sklearn.preprocessing import StandardScaler
sc_x=StandardScaler()
x_train=sc_x.fit_transform(x_tr)
x_test=sc_x.transform(x_ts)
sc_y = StandardScaler()
y_train = sc_y.fit_transform(y_tr.reshape(-1,1))
y_test=sc_y.transform(y_ts.reshape(-1,1))

### Fitting multiple linear regression
from sklearn.linear_model import LinearRegression
regressor=LinearRegression()
regressor.fit(x_train,y_train)

# evaluating predictoin
y_pr=regressor.predict(x_ts)
y_ts=y_ts.reshape(-1,1)
Error_nonscaling=sum(y_pr**2-y_ts**2)

y_prediction=regressor.predict(x_test)
Error_scaling=sum(y_prediction**2-y_test**2)

############################### EXAMPLE OF PREDICTION ###################################
X=[[0,1,100000,150000,300000],[1,0,120000,140000,400000],[0,0,150000,160000,140000]]
X=sc_x.transform(X)
Y=regressor.predict(X)
Y=sc_y.inverse_transform(Y)

X=[[0,1,100000,150000,300000],[1,0,120000,140000,400000],[0,0,150000,160000,140000]]
y=regressor.predict(X)

'''



# number 2

# importing librries
import numpy as np
import pandas as pd

### importing dataset
data=pd.read_csv('50_Startups.csv')
x=data.iloc[:,:-1].values
y=data.iloc[:,-1].values

### preprocessing

# Encoding categorical data
from sklearn.preprocessing import LabelEncoder, OneHotEncoder
from sklearn.compose import ColumnTransformer

lableencoder=LabelEncoder()
x[:,3]=lableencoder.fit_transform(x[:,3])

CT=ColumnTransformer([("Country",OneHotEncoder(),[3])],remainder='passthrough')
x=CT.fit_transform(x)

# Avoiding the Dummy variables Trap
x=x[:,1:]

# feature scaling
from sklearn.preprocessing import StandardScaler
sc_x=StandardScaler()
x=sc_x.fit_transform(x)

sc_y = StandardScaler()
y= sc_y.fit_transform(y.reshape(-1,1))

### Building the OPTIMAL model using Feature Selection (Backward Elimination)
# this didn't work for me: "import statsmodels.formula.api as sm"
import statsmodels.api as sm
x=np.append(arr=np.ones((50,1)).astype(int),values=x,axis=1)
#1
x_opt=np.array(x[:,[0,1,2,3,4,5]],dtype=float)
regressor_OLS=sm.OLS(endog=y,exog=x_opt).fit()
regressor_OLS.summary()
#2
x_opt=np.array(x[:,[1,2,3,4,5]],dtype=float)
regressor_OLS=sm.OLS(endog=y,exog=x_opt).fit()
regressor_OLS.summary()
#3
x_opt=np.array(x[:,[1,3,4,5]],dtype=float)
regressor_OLS=sm.OLS(endog=y,exog=x_opt).fit()
regressor_OLS.summary()
#4
x_opt=np.array(x[:,[3,4,5]],dtype=float)
regressor_OLS=sm.OLS(endog=y,exog=x_opt).fit()
regressor_OLS.summary()
#5
x_opt=np.array(x[:,[3,5]],dtype=float)
regressor_OLS=sm.OLS(endog=y,exog=x_opt).fit()
regressor_OLS.summary()
#6
x_opt=np.array(x[:,[3]],dtype=float)
regressor_OLS=sm.OLS(endog=y,exog=x_opt).fit()
regressor_OLS.summary()