# Importing the libraries
import numpy as np
import pandas as pd
import matplotlib.pyplot as plt
import seaborn as sns

# Loading the dataset
dt = pd.read_csv("housing.csv")

# Getting rid of Outliers using Z-score method
from scipy import stats
z = np.abs(stats.zscore(dt["total_rooms"]))
dt['z_score']=z
dt = dt[dt['z_score']<3]

# Encoding categorical data
dt = pd.concat([dt, pd.get_dummies(dt['ocean_proximity'], prefix='ocean_prox',
                                   drop_first=True)], axis=1)

# Data Imputation
g = dt.groupby('ocean_proximity')
dt['total_bedrooms'] = g['total_bedrooms'].apply(lambda x: x.fillna(x.median()))

dt['rooms/households'] = dt['total_rooms']/dt['households']
dt['bedrooms/households'] = dt['total_bedrooms']/dt['households']

dt.drop(['ocean_proximity', 'total_rooms', 'total_bedrooms', 'z_score'], 
        axis=1, inplace=True)

# Splitting the dataset into X and y values
X = dt
X = X.drop(['median_house_value'], axis=1)
y = dt['median_house_value']

# Splitting the dataset into the Training set and Test set
from sklearn.model_selection import train_test_split
X_train, X_test, y_train, y_test = train_test_split(X, y, test_size = 0.2, 
                                                    random_state = 42)

# Feature Scaling 
from sklearn.preprocessing import StandardScaler
sc_X = StandardScaler()
X_train = sc_X.fit_transform(X_train)
X_test = sc_X.transform(X_test)

# Fitting multiple linear regression to the training set
from sklearn.linear_model import LinearRegression
regressor = LinearRegression(n_jobs=-1)
regressor.fit(X_train, y_train)

# Predicting the test set results
y_pred = regressor.predict(X_test)