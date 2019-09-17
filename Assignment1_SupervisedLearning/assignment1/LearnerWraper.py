## Name: Yiying Zhang  GTID: yzhang3006.
## Assignmenet 1: Supervised Learning

import numpy as np
import pandas as pd
from sklearn.model_selection import cross_val_score, GridSearchCV, cross_validate, train_test_split
from sklearn import tree
from sklearn.metrics import accuracy_score, classification_report
from sklearn.svm import SVC
from sklearn.linear_model import LinearRegression
from sklearn.ensemble import RandomForestClassifier
from sklearn.preprocessing import StandardScaler, normalize
from sklearn.decomposition import PCA


######################################### Reading and Cleaning the Data ###############################################
# data resource: https://archive.ics.uci.edu/ml/datasets/Online+Shoppers+Purchasing+Intention+Dataset
data = pd.read_csv('online_shoppers_intention.csv')

#Data cleaning: categorical data must be encoded in ML method like: Linear regression, SVM, Neural network.
data[['Revenue', 'Weekend']] = (data[['Revenue', 'Weekend']] == True).astype(int)
VisitorType_map = {'VisitorType': {"Returning_Visitor": 1, "Other": 2, "New_Visitor": 0}}
data.replace(VisitorType_map, inplace= True)
Month_map =  {'Jan':1, 'Feb':2, 'Mar':3, 'Apr':4, 'May': 5,'June': 6, 'Jul':7, 'Aug':8, 'Sep':9, 'Oct':10, 'Nov': 11, 'Dec': 12}
data.replace(Month_map, inplace= True)

# Separate out the x_data and y_data.
x_data = data.loc[:, data.columns != "Revenue"]
y_data = data.loc[:, "Revenue"]
print("after-cleaning data", data.head())

# The random state to use while splitting the data.
random_state = 100

# XXX
# Split 70% of the data into training and 30% into test sets. Call them x_train, x_test, y_train and y_test.
# Use the train_test_split method in sklearn with the parameter 'shuffle' set to true and the 'random_state' set to 100.
# XXX
xTrain, xTest, yTrain, yTest = train_test_split(x_data, y_data, train_size=0.7, random_state=random_state)


# ############################################### Decision Tree ###################################################
# Create a DT classifier and train it.

DecisionTree = tree.DecisionTreeClassifier()
DecisionTree.fit(xTrain, yTrain)

# Test its accuracy (on the training set) using the accuracy_score method.
# Test its accuracy (on the testing set) using the accuracy_score method.
# Note: Round the output values greater than or equal to 0.5 to 1 and those less than 0.5 to 0. You can use y_predict.round() or any other method.

trainScore = accuracy_score(yTrain, DecisionTree.predict(xTrain).round())
testScore = accuracy_score(yTest, DecisionTree.predict(xTest).round())
print('trainScore', trainScore)
print('testScoret', testScore)
