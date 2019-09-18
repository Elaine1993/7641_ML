## Name: Yiying Zhang  GTID: yzhang3006.
## Assignmenet 1: Supervised Learning

import numpy as np
import pandas as pd
from sklearn.model_selection import cross_val_score, GridSearchCV, cross_validate, train_test_split
from sklearn import tree
from sklearn.metrics import accuracy_score, classification_report
from sklearn.svm import SVC
from sklearn.neighbors import KNeighborsClassifier
from sklearn.ensemble import AdaBoostClassifier
from sklearn.preprocessing import StandardScaler, normalize
from sklearn.neural_network import MLPClassifier
from sklearn.decomposition import PCA
import matplotlib.pyplot as plt


######################################### Reading and Cleaning the Data ###############################################
# data resource1: https://archive.ics.uci.edu/ml/datasets/Online+Shoppers+Purchasing+Intention+Dataset
data1 = pd.read_csv('online_shoppers_intention.csv')

#Data cleaning: categorical data must be encoded in ML method like: Linear regression, SVM, Neural network.
data1[['Revenue', 'Weekend']] = (data1[['Revenue', 'Weekend']] == True).astype(int)
VisitorType_map = {'VisitorType': {"Returning_Visitor": 1, "Other": 2, "New_Visitor": 0}}
data1.replace(VisitorType_map, inplace= True)
Month_map =  {'Jan':1, 'Feb':2, 'Mar':3, 'Apr':4, 'May': 5,'June': 6, 'Jul':7, 'Aug':8, 'Sep':9, 'Oct':10, 'Nov': 11, 'Dec': 12}
data1.replace(Month_map, inplace= True)

# Separate out the x_data and y_data.
x_data1 = data1.loc[:, data1.columns != "Revenue"]
y_data1 = data1.loc[:, "Revenue"]
print("after-cleaning data", data1.head())

# data resource2: https://archive.ics.uci.edu/ml/datasets/default+of+credit+card+clients
data2 = pd.read_excel("default_of_credit_card_clients.xls", skiprows=[1], index_col= 0)
print("before-cleaning data2\n", data2.head())

# Separate out the x_data and y_data.
x_data2 = data2.loc[:, data2.columns != "Y"]
y_data2 = data2.loc[:, "Y"]
#print("before-cleaning data2", data2.head())



# ############################################### Helper Functions ###################################################
# The random state to use while splitting the data.
random_state = 100

####Split i% of the data into training and 1-i% into test sets. #######
# Use the train_test_split method in sklearn with the parameter 'shuffle' set to true and the 'random_state' set to 100.
def splitData(xdata, ydata, trainsize):
     xTrain, xTest, yTrain, yTest = train_test_split(xdata, ydata, train_size = trainsize, random_state=random_state)
     return xTrain, xTest, yTrain, yTest

####### Plot: train size vs accuracy #############
# call after iteration is done and get data list
def plotAccuray_TrainSize(TrainResult,TestResult):
    xrange = np.arange(0.1,1.0, 0.1)
    #axes = plt.gca() #"get the current axes"
    #axes.set_ylim([0,1.2]) # set current axes:
    plt.plot(xrange,TrainResult, xrange, TestResult)
    plt.show()

####### experiment: iteration for n time, each time, use change train size ####
def experiment(n, f):
    TrainResult1 = []
    TestResult1 = []
    TrainResult2 = []
    TestResult2 = []
    for i in range(1, n):
        a = i / n
        print("SVM Training ", a)
        trainScore1, testScore1 = f(x_data1, y_data1, a)
        TrainResult1.append(trainScore1)
        TestResult1.append(testScore1)
        trainScore2, testScore2 = f(x_data2, y_data2, a)
        TrainResult2.append(trainScore2)
        TestResult2.append(testScore2)
    return TrainResult1, TestResult1, TrainResult2, TestResult2

# ############################################### Decision Tree ###################################################
def DTLearner(xdata, ydata, trainsize):
    xTrain, xTest, yTrain, yTest =  splitData(xdata, ydata, trainsize)
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
    return trainScore, testScore

# ############################################### Decision Tree Experiment with 2 data_sets & plot ###################################################
DT_TrainResult1, DT_TestResult1, DT_TrainResult2, DT_TestResult2 = experiment(10, DTLearner)
plotAccuray_TrainSize(DT_TrainResult1, DT_TestResult1)
plotAccuray_TrainSize(DT_TrainResult2, DT_TestResult2)



# ############################################ Support Vector Machine ###################################################
# XXX
# TODO: Pre-process the data to standardize or normalize it, otherwise the grid search will take much longer
# TODO: Create a SVC classifier and train it.
# XXX
def SVMLearner(xdata, ydata, trainsize):
    xTrain, xTest, yTrain, yTest = train_test_split(xdata, ydata, train_size=trainsize, random_state=random_state)
    s = StandardScaler().fit(xTrain)
    xTrainSVC = s.transform(xTrain)
    xTestSVC = s.transform(xTest)
    svc = SVC().fit(xTrainSVC, yTrain)
    trainScore = accuracy_score(yTrain, svc.predict(xTrainSVC).round())
    testScore = accuracy_score(yTest, svc.predict(xTestSVC).round())
    print('SVM: trainScore', trainScore)
    print('SVM: testScoret', testScore)
    return trainScore, testScore
# ###################################### Support Vector Machine Experiment with 2 dataset & plot ############################################
# SVM_TrainResult1,SVM_TestResult1,SVM_TrainResult2,SVM_TestResult2 = experiment(10, SVMLearner)
# plotAccuray_TrainSize(SVM_TrainResult1, SVM_TestResult1)
# plotAccuray_TrainSize(SVM_TrainResult2, SVM_TestResult2)


# ############################################### K Nearest Neighbor(KNN) ###################################################
def KNNLearner(xdata, ydata, trainsize, k=3):
    xTrain, xTest, yTrain, yTest = train_test_split(xdata, ydata, train_size=trainsize, random_state=random_state)
    knn = KNeighborsClassifier(n_neighbors= k).fit(xTrain, yTrain)
    trainScore = accuracy_score(yTrain, knn.predict(xTrain).round())
    testScore = accuracy_score(yTest, knn.predict(xTest).round())
    print('KNN: trainScore', trainScore)
    print('KNN: testScoret', testScore)
    return trainScore, testScore
# ###################################### KNN Experiment with 2 dataset & plot ############################################
# knn_TrainResult1, knn_TestResult1, knn_TrainResult2, knn_TestResult2 = experiment(10, KNNLearner)
# plotAccuray_TrainSize(knn_TrainResult1, knn_TestResult1)
# plotAccuray_TrainSize(knn_TrainResult2, knn_TestResult2)

# ############################################### Boosting: AdaBoost ###################################################
def BoostLearner(xdata, ydata, trainsize):
    xTrain, xTest, yTrain, yTest = train_test_split(xdata, ydata, train_size=trainsize, random_state=random_state)
    adb = AdaBoostClassifier(base_estimator= None).fit(xTrain, yTrain) #default is decision tree
    trainScore = accuracy_score(yTrain, adb.predict(xTrain).round())
    testScore = accuracy_score(yTest, adb.predict(xTest).round())
    print('Boosting: trainScore', trainScore)
    print('Boosting: testScoret', testScore)
    return trainScore, testScore
# ###################################### Boosting Experiment with 2 dataset & plot ############################################
# boost_TrainResult1, boost_TestResult1, boost_TrainResult2, boost_TestResult2 = experiment(10, BoostLearner)
# plotAccuray_TrainSize(boost_TrainResult1, boost_TestResult1)
# plotAccuray_TrainSize(boost_TrainResult2, boost_TestResult2)

# ############################################### Neural Network #######################################################
def NeuralNetworkLearner(xdata, ydata, trainsize):
    xTrain, xTest, yTrain, yTest = train_test_split(xdata, ydata, train_size=trainsize, random_state=random_state)
    nn = MLPClassifier(activation = 'logistic', solver = 'sgd', hidden_layer_sizes=(10,15), learning_rate_init= 0.001).fit(xTrain, yTrain) #training time cooresponding to hidden layer size
    trainScore = accuracy_score(yTrain, nn.predict(xTrain).round())
    testScore = accuracy_score(yTest, nn.predict(xTest).round())
    print('Neural Network: trainScore', trainScore)
    print('Neural Network: testScoret', testScore)
    return trainScore, testScore
# ###################################### Neural Network Experiment with 2 dataset & plot ############################################
# nn_TrainResult1, nn_TestResult1, nn_TrainResult2, nn_TestResult2 = experiment(10, NeuralNetworkLearner)
# plotAccuray_TrainSize(nn_TrainResult1, nn_TestResult1)
# plotAccuray_TrainSize(nn_TrainResult2, nn_TestResult2)
