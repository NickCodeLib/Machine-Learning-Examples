# -*- coding: utf-8 -*-
"""
Created on Wed Apr 12 09:51:12 2023

@author: he587e
"""

#Importing libraries
import numpy as np
import pandas as pd
import matplotlib.pyplot as plt
%matplotlib inline
import seaborn as sns
from sklearn.linear_model import LinearRegression
from sklearn.datasets import make_regression
from sklearn.model_selection import train_test_split

#Creating sample datasets
X,Y = make_regression(n_features = 1, noise = 5, n_samples = 5000)
plt.xlabel('X')
plt.ylabel('Y')
plt.scatter(X,Y)

#Making a linear model and getting coefficients
linear_model = LinearRegression()
linear_model.fit(X,Y)

linear_model.coef_
linear_model.intercept_

#Using trained model to making predictions of the X values that we generated above
pred = linear_model.predict(X)
plt.scatter(X,Y, label = "training")
plt.scatter(X,pred, label = "prediction")
plt.xlabel('X')
plt.ylabel('Y')
plt.legend()
plt.show()

#Creating a second model based off of actual data -----------------------------------------------------------------------------------------------------
url = 'https://raw.githubusercontent.com/Apress/supervised-learning-w-python/master/Chapter2/House_data_LR.csv'
house_df = pd.read_csv(url)
house_df.drop(columns = 'Unnamed: 0', inplace = True)

X = house_df.drop(columns = 'price')
y = house_df.drop(columns = 'sqft_living')

#Splitting model into train and test and fitting the model 
X_train, X_test, y_train, y_test = train_test_split(X, y, test_size = 0.25, random_state = 5)

linear_regression = LinearRegression()
linear_regression.fit(X_train,y_train)

y_pred = linear_regression.predict(X_test)

#Graphing our results for training dataset
plt.scatter(X_train, y_train, color = 'r')
plt.plot(X_train, linear_regression.predict(X_train), color = 'b')
plt.title('Sqft Living vs Price: Training')
plt.xlabel("Square Feet")
plt.ylabel('House Price')
plt.show()

#Graphing our results on the testing data
plt.scatter(X_test, y_test, color = 'r')
plt.plot(X_train, linear_regression.predict(X_train), color = 'b')
plt.title('Sqft Living vs Price for: Test')
plt.xlabel('Square feet')
plt.ylabel('House Price')

#Calculating Metrics to see how well our model preformed 
from sklearn.metrics import mean_squared_error
from math import sqrt
from sklearn.metrics import r2_score
rmse = sqrt(mean_squared_error(y_test, y_pred))
r2 = r2_score(y_test, y_pred)
print(rmse, r2)