#!/usr/bin/env python
# coding: utf-8

# In[2]:


import pandas as pd
from sklearn.model_selection import train_test_split
from sklearn.preprocessing import StandardScaler 
from sklearn.linear_model import LinearRegression
from sklearn.ensemble import RandomForestRegressor
import numpy as np

def custom_accuracy(y_test,y_pred,thresold):
    right = 0
    l = len(y_pred)
    for i in range(0,l):
        if(abs(y_pred[i]-y_test[i]) <= thresold):
            right += 1
    return ((right/l)*100)

lin = LinearRegression()
sc = StandardScaler()
dataset = pd.read_csv('/Users/sushruth/Desktop/CricketScorePredictor-master/data/t20.csv')
X = dataset.iloc[:,[7,8,9,12,13]].values
y = dataset.iloc[:,14].values
X_train, X_test, y_train, y_test = train_test_split(X, y, test_size = 0.25, random_state = 0)
X_train = sc.fit_transform(X_train)
X_test = sc.transform(X_test)
lin.fit(X_train, y_train)
lin = RandomForestRegressor(n_estimators = 100, max_features = None)
lin.fit(X_train, y_train)
y_pred = lin.predict(X_test)
score = lin.score(X_test, y_test)*100
print("R-squared value:" , score)
print("Custom accuracy:" , custom_accuracy(y_test,y_pred,20))
new_prediction = lin.predict(sc.transform(np.array([[163, 1,23,90,2 ]]))) # current total, wickets, overs, striker, nonstriker
print(new_prediction)


# In[ ]:




