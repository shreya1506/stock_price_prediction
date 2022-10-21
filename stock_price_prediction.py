# -*- coding: utf-8 -*-
"""
Created on Sun Aug 28 13:44:06 2022

@author: abhishek
"""

"""
1. Importing the Required Libraries
"""

import numpy as np
import matplotlib.pyplot as plt
import pandas as pd
from sklearn import preprocessing
from sklearn import metrics
from sklearn.model_selection import train_test_split
from sklearn.linear_model import LinearRegression


"""
2. Data Preparation And Visualization
"""

df = pd.read_csv("F:/ML-DS-PYTHON/My_projects/Stock_Price_Prediction/dataset/TSLA.csv")
df.head()
df.info()
df.describe()


"""
3. Splitting Data In X = Features and Y = Labeles
"""

X = df[['High','Low','Open','Volume']].values
Y = df['Close'].values
print(X)
print(Y)


"""
4. Train and Test Split
"""

# Split data into testing and training sets
X_train, X_test, Y_train, Y_test = train_test_split(X,Y, test_size = 0.3, random_state = 1)


"""
5. Training the Model
"""

#from sklearn.linear_model import LinearRegression
# Create Regression Model 
Model = LinearRegression()

# Train the model
Model.fit(X_train, Y_train)
#Printing Coefficient
print(Model.coef_)

# Use model to make predictions
predicted = Model.predict(X_test) 
print(predicted)

"""
6. Combining The Actual and Predicted data to match
"""

df_1 = pd.DataFrame({'Actual': Y_test.flatten(), 'Predicted' : predicted.flatten()})
df_1.head(20)

"""
7. Validating the Fit
"""
import math
print('Mean Absolute Error:', metrics.mean_absolute_error(Y_test,predicted))
print('Mean Squared Error:', metrics.mean_squared_error(Y_test,predicted))
print('Root Mean Squared Error:', math.sqrt(metrics.mean_squared_error(Y_test,predicted)))

'Graph ploting'
graph = df_1.head(20)
graph.plot(kind='bar')