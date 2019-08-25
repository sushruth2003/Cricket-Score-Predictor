import pandas as pd
from sklearn.model_selection import train_test_split
from sklearn.preprocessing import StandardScaler
from sklearn.linear_model import LinearRegression
from sklearn.ensemble import RandomForestRegressor
import numpy as np
import seaborn as sns


def custom_accuracy(y_test,y_pred,thresold):
    right = 0
    l = len(y_pred)
    for i in range(0,l):
        if(abs(y_pred[i]-y_test[i]) <= thresold):
            right += 1
    return ((right/l)*100)

lin = LinearRegression()
sc = StandardScaler()
dataset = pd.read_csv('/Users/sushruth/Desktop/CricketScorePredictor-master/data/ipl.csv')
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
#new_prediction1 = lin.predict(sc.transform(np.array([[37, 0,4,14,22 ]])))
#new_prediction2 = lin.predict(sc.transform(np.array([[58, 2,9,5,7 ]])))
#new_prediction3 = lin.predict(sc.transform(np.array([[94, 4,14,2,23 ]]))) # current total, wickets, overs, striker, nonstriker
#new_prediction4 = lin.predict(sc.transform(np.array([[120, 5,17,24,4 ]])))
#print("4 overs", new_prediction1)
#print("9 overs", new_prediction2)
#print("14 overs", new_prediction3)
#print("17 overs", new_prediction4)
run_total = int(input("How many runs have been scored: "))
wicket_total = int(input("How many wickets have fallen: "))
overs = int(input("How many overs have passed: "))
striker = int(input("How many runs does the striker have: "))
nonstriker = int(input("How many runs does the nonstriker have: "))
new_prediction  = lin.predict(sc.transform(np.array([[run_total,wicket_total, overs, striker, nonstriker]])))
print("Based on past ipl data for batting first your total is", new_prediction)
