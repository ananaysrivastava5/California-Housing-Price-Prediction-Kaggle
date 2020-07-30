# -*- coding: utf-8 -*-
"""
Created on Thu Jul 23 22:06:06 2020

@author: Ananay Srivastava
"""

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

'''
from sklearn.model_selection import RandomizedSearchCV
n_estimators = [int(x) for x in np.linspace(start = 200, stop = 2000, num = 10)]
max_features = ['auto', 'sqrt']
max_depth = [int(x) for x in np.linspace(5, 110, num = 5)]
max_depth.append(None)
min_samples_split = np.linspace(0.1, 1.0, 10, endpoint=True) #[2, 5, 10]
min_samples_leaf = np.linspace(0.1, 0.5, 5, endpoint=True)  #[1, 2, 4]
bootstrap = [True, False]
# Create the random grid
random_grid = {'n_estimators': n_estimators,
               'max_features': max_features,
               'max_depth': max_depth,
               'min_samples_split': min_samples_split,
               'min_samples_leaf': min_samples_leaf,
               'bootstrap': bootstrap}

# Use the random grid to search for best hyperparameters
# First create the base model to tune
from sklearn.ensemble import RandomForestRegressor
rf = RandomForestRegressor()
rf_random = RandomizedSearchCV(estimator = rf, param_distributions = random_grid,
                               n_iter = 300, cv = 5, verbose=2, random_state=42, 
                               n_jobs = -1)
model_results = rf_random.fit(X_train, y_train)

# Model best results
print(model_results.best_params_)
'''
# Fitting multiple Decision Tree Regression to the training set
from sklearn.ensemble import RandomForestRegressor
regressor = RandomForestRegressor(n_estimators= 2000, random_state = 42)
regressor.fit(X_train, y_train)

# Predicting the test set results
y_pred = regressor.predict(X_test)

# Root Mean Squared Error (RMSE) = 48,248.382
from sklearn import metrics
lr_rmse = np.sqrt(metrics.mean_squared_error(y_test, y_pred))