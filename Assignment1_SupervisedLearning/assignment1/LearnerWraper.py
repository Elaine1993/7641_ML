## Name: Yiying Zhang  GTID: yzhang3006.
## Assignmenet 1: Supervised Learning

import numpy as np
import pandas as pd
from sklearn.model_selection import cross_val_score, cross_validate, train_test_split
from sklearn import tree
from sklearn.metrics import accuracy_score
from sklearn.svm import SVC
from sklearn.neighbors import KNeighborsClassifier
from sklearn.ensemble import AdaBoostClassifier
from sklearn.preprocessing import StandardScaler
from sklearn.neural_network import MLPClassifier
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

#######Split i% of the data into training and 1-i% into test sets. #######
# Use the train_test_split method in sklearn with the parameter 'shuffle' set to true and the 'random_state' set to 100.
def fixedsplitData(xdata, ydata):
     xTrain, xTest, yTrain, yTest = train_test_split(xdata, ydata, train_size = 0.7, random_state=random_state)
     return xTrain, xTest, yTrain, yTest

######### experiment_size: iteration for n time, each time, use change train size ########
def experiment_size(n, f):
    TrainResult1 = []
    TestResult1 = []
    TrainResult2 = []
    TestResult2 = []
    for i in range(1, n):
        a = i / n
        trainScore1, testScore1 = f(x_data1, y_data1, a)
        TrainResult1.append(trainScore1)
        TestResult1.append(testScore1)
        trainScore2, testScore2 = f(x_data2, y_data2, a)
        TrainResult2.append(trainScore2)
        TestResult2.append(testScore2)
    print("TrainResult1", TrainResult1)
    print("TestResult1", TestResult1)
    print("TrainResult2", TrainResult2)
    print("TestResult2", TestResult2)
    print("\n")
    return TrainResult1, TestResult1, TrainResult2, TestResult2

######### experiment_parameter: iteration for n time, each time, change parameter########
def experiment_parameter(n, f):
    xTrain1, xTest1, yTrain1, yTest1= fixedsplitData(x_data1, y_data1)
    xTrain2, xTest2, yTrain2, yTest2= fixedsplitData(x_data2, y_data2)
    TrainResult_parameter1 = []
    TestResult_parameter1 = []
    TrainResult_parameter2 = []
    TestResult_parameter2 = []
    for i in range(1, n):
        trainScore1, testScore1 = f(xTrain1, xTest1, yTrain1, yTest1, i)
        TrainResult_parameter1.append(trainScore1)
        TestResult_parameter1.append(testScore1)
        trainScore2, testScore2 = f(xTrain2, xTest2, yTrain2, yTest2, i)
        TrainResult_parameter2.append(trainScore2)
        TestResult_parameter2.append(testScore2)
    print("TrainResult_parameter1 =", TrainResult_parameter1)
    print("TestResult_parameter1 =", TestResult_parameter1)
    print("TrainResult_parameter2 =", TrainResult_parameter2)
    print("TestResult_parameter2=", TestResult_parameter2)
    print("\n")
    return TrainResult_parameter1, TestResult_parameter1, TrainResult_parameter2, TestResult_parameter2


######### Plot: train size vs accuracy #############
# call after iteration is done and get data list
def plotAccuray_TrainSize(TrainResult1,TestResult1, TrainResult2,TestResult2,plotname, parameter_name):
    xrange = np.arange(0.01,1.0, 0.01)
    plt.plot(xrange, TrainResult1, label="TrainAccuracy-Dataset1")
    plt.plot(xrange, TestResult1, label="TestAccuracy-Dataset1")
    plt.plot(xrange, TrainResult2, label="TrainAccuracy-Dataset2")
    plt.plot(xrange, TestResult2, label="TestAccuracy-Dataset2")
    plt.legend()
    plt.xlabel(parameter_name)
    plt.ylabel("Accuracy")
    plt.title("Accuracy Score of " + str(plotname))
    plt.savefig("Accuracy Score of " + str(plotname) + ".png")
    plt.close()

def plotAccuray_parameter(TrainResult1,TestResult1, TrainResult2,TestResult2,plotname, parameter_name, n):
    xrange = np.arange(1,n)
    plt.plot(xrange, TrainResult1, label="TrainAccuracy-Dataset1")
    plt.plot(xrange, TestResult1, label="TestAccuracy-Dataset1")
    plt.plot(xrange, TrainResult2, label="TrainAccuracy-Dataset2")
    plt.plot(xrange, TestResult2, label="TestAccuracy-Dataset2")
    plt.legend()
    plt.xlabel(parameter_name)
    plt.ylabel("Accuracy")
    plt.title("Accuracy Score of " + str(plotname))
    plt.savefig("Accuracy Score of " + str(plotname) + ".png")
    plt.close()


# ############################################### Decision Tree ###################################################
def DTLearner(xdata, ydata, trainsize):
    xTrain, xTest, yTrain, yTest = train_test_split(xdata, ydata, train_size=trainsize, random_state=random_state)
    # Create a DT classifier and train it.
    dt = tree.DecisionTreeClassifier().fit(xTrain, yTrain)
    # Note: Round the output values greater than or equal to 0.5 to 1 and those less than 0.5 to 0. You can use y_predict.round() or any other method.
    trainScore = accuracy_score(yTrain, dt.predict(xTrain).round())
    testScore = accuracy_score(yTest, dt.predict(xTest).round())
    # print('trainScore', trainScore)
    # print('testScoret', testScore)
    return trainScore, testScore
def DTLearner_parameter(xTrain, xTest, yTrain, yTest, p):
    # Create a DT classifier and train it.
    dt = tree.DecisionTreeClassifier(max_depth = p).fit(xTrain, yTrain)
    trainScore = accuracy_score(yTrain, dt.predict(xTrain).round())
    testScore = accuracy_score(yTest, dt.predict(xTest).round())
    print('trainScore', trainScore)
    print('testScoret', testScore)
    return trainScore, testScore

# ############################################### Decision Tree Experiment with 2 data_sets & plot ###################################################
DT_TrainResult1, DT_TestResult1, DT_TrainResult2, DT_TestResult2 = experiment_size(100, DTLearner)
plotAccuray_TrainSize(DT_TrainResult1, DT_TestResult1,DT_TrainResult2, DT_TestResult2, "DecisionTree(TrainingSize)", "Training size(1%~99%)")
DT_TrainResult_p1, DT_TestResult_p1, DT_TrainResult_p2, DT_TestResult_p2 = experiment_parameter(100, DTLearner_parameter)
plotAccuray_parameter(DT_TrainResult_p1, DT_TestResult_p1,DT_TrainResult_p2, DT_TestResult_p2, "DecisionTree(parameter)", "Parameter:max_depth", 100)

# ############################################ Support Vector Machine ###################################################
# XXX
# Pre-process the data to standardize or normalize it, otherwise the grid search will take much longer
# Create a SVC classifier and train it.
# XXX
def SVMLearner(xdata, ydata, trainsize):
    xTrain, xTest, yTrain, yTest = train_test_split(xdata, ydata, train_size=trainsize, random_state=random_state)
    s = StandardScaler().fit(xTrain)
    xTrainSVC = s.transform(xTrain)
    xTestSVC = s.transform(xTest)
    svc = SVC().fit(xTrainSVC, yTrain)
    trainScore = accuracy_score(yTrain, svc.predict(xTrainSVC).round())
    testScore = accuracy_score(yTest, svc.predict(xTestSVC).round())
    # print('SVM: trainScore', trainScore)
    # print('SVM: testScoret', testScore)
    return trainScore, testScore

def SVMLearner_parameter(xTrain, xTest, yTrain, yTest, p):
    if(p > 4):
        return
    kernelList= ['rbf','linear', 'poly', 'sigmoid']
    s = StandardScaler().fit(xTrain)
    xTrainSVC = s.transform(xTrain)
    xTestSVC = s.transform(xTest)
    svc = SVC(kernel= kernelList[p-1]).fit(xTrainSVC, yTrain)
    trainScore = accuracy_score(yTrain, svc.predict(xTrainSVC).round())
    testScore = accuracy_score(yTest, svc.predict(xTestSVC).round())
    print('SVM: trainScore', trainScore)
    print('SVM: testScoret', testScore)
    return trainScore, testScore
def SVM_plot_kernel(SVM_TrainResult_p1,SVM_TestResult_p1,SVM_TrainResult_p2,SVM_TestResult_p2):
    xrange = ['rbf', 'linear', 'poly', 'sigmoid']
    plt.scatter(xrange, SVM_TrainResult_p1, label="TrainAccuracy-Dataset1")
    plt.scatter(xrange, SVM_TestResult_p1, label="TestAccuracy-Dataset1")
    plt.scatter(xrange, SVM_TrainResult_p2, label="TrainAccuracy-Dataset2")
    plt.scatter(xrange, SVM_TestResult_p2, label="TestAccuracy-Dataset2")
    plt.legend()
    plt.xlabel("Parameter: kernal")
    plt.ylabel("Accuracy")
    plt.title("Accuracy Score of Parameter: kernal")
    plt.savefig("Accuracy Score of Parameter: kernal.png")
    plt.show()


# ###################################### Support Vector Machine Experiment with 2 dataset & plot ############################################
SVM_TrainResult1,SVM_TestResult1,SVM_TrainResult2,SVM_TestResult2 = experiment_size(100, SVMLearner)
plotAccuray_TrainSize(SVM_TrainResult1, SVM_TestResult1,SVM_TrainResult2, SVM_TestResult2, "SVM", "Training size(1%~99%)")
SVM_TrainResult_p1,SVM_TestResult_p1,SVM_TrainResult_p2,SVM_TestResult_p2 = experiment_parameter(5, SVMLearner_parameter)
SVM_plot_kernel(SVM_TrainResult_p1,SVM_TestResult_p1,SVM_TrainResult_p2,SVM_TestResult_p2)


# ############################################### K Nearest Neighbor(KNN) ###################################################
def KNNLearner(xdata, ydata, trainsize ):
    xTrain, xTest, yTrain, yTest = train_test_split(xdata, ydata, train_size=trainsize, random_state=random_state)
    knn = KNeighborsClassifier().fit(xTrain, yTrain)
    trainScore = accuracy_score(yTrain, knn.predict(xTrain).round())
    testScore = accuracy_score(yTest, knn.predict(xTest).round())
    # print('KNN: trainScore', trainScore)
    # print('KNN: testScoret', testScore)
    return trainScore, testScore

def KNNLearner_parameter(xTrain, xTest, yTrain, yTest, p):
    knn = KNeighborsClassifier(n_neighbors= p).fit(xTrain, yTrain)
    trainScore = accuracy_score(yTrain, knn.predict(xTrain).round())
    testScore = accuracy_score(yTest, knn.predict(xTest).round())
    # print('KNN: trainScore', trainScore)
    # print('KNN: testScoret', testScore)
    return trainScore, testScore
# ###################################### KNN Experiment with 2 dataset & plot ############################################
knn_TrainResult1, knn_TestResult1, knn_TrainResult2, knn_TestResult2 = experiment_size(100, KNNLearner)
plotAccuray_TrainSize(knn_TrainResult1, knn_TestResult1,knn_TrainResult2, knn_TestResult2, "KNN",  "Training size(1%~99%)")
knn_TrainResult_p1, knn_TestResult_p1, knn_TrainResult_p2, knn_TestResult_p2 = experiment_parameter(50, KNNLearner_parameter)
plotAccuray_parameter(knn_TrainResult_p1, knn_TestResult_p1, knn_TrainResult_p2, knn_TestResult_p2, "KNN(parameter)", "Parameter: k", 50)

# ############################################### Boosting: AdaBoost ###################################################
def BoostLearner(xdata, ydata, trainsize):
    xTrain, xTest, yTrain, yTest = train_test_split(xdata, ydata, train_size=trainsize, random_state=random_state)
    adb = AdaBoostClassifier(n_estimators= None).fit(xTrain, yTrain) #default is decision tree, n_estimators  = 50
    trainScore = accuracy_score(yTrain, adb.predict(xTrain).round())
    testScore = accuracy_score(yTest, adb.predict(xTest).round())
    # print('Boosting: trainScore', trainScore)
    # print('Boosting: testScoret', testScore)
    return trainScore, testScore
def BoostLearner_paramter(xTrain, xTest, yTrain, yTest, p):
    adb = AdaBoostClassifier(n_estimators = p).fit(xTrain, yTrain) #default is decision tree
    trainScore = accuracy_score(yTrain, adb.predict(xTrain).round())
    testScore = accuracy_score(yTest, adb.predict(xTest).round())
    # print('Boosting: trainScore', trainScore)
    # print('Boosting: testScoret', testScore)
    return trainScore, testScore
# ###################################### Boosting Experiment with 2 dataset & plot ############################################
boost_TrainResult1, boost_TestResult1, boost_TrainResult2, boost_TestResult2 = experiment_size(100, BoostLearner)
plotAccuray_TrainSize(boost_TrainResult1, boost_TestResult1, boost_TrainResult2, boost_TestResult2, "Boosting", "Training size(1%~99%)")
boost_TrainResult_p1, boost_TestResult_p1, boost_TrainResult_p2, boost_TestResult_p2 = experiment_parameter(100, BoostLearner_paramter)
plotAccuray_parameter(boost_TrainResult_p1, boost_TestResult_p1, boost_TrainResult_p2, boost_TestResult_p2, "Boosting(parameter)", "Parameter: n_estimators", 100)


# ############################################### Neural Network #######################################################
def NeuralNetworkLearner(xdata, ydata, trainsize):
    xTrain, xTest, yTrain, yTest = train_test_split(xdata, ydata, train_size=trainsize, random_state=random_state)
    nn = MLPClassifier(activation = 'logistic', solver = 'sgd', hidden_layer_sizes=(10,15), learning_rate_init= 0.001).fit(xTrain, yTrain) #training time cooresponding to hidden layer size
    trainScore = accuracy_score(yTrain, nn.predict(xTrain).round())
    testScore = accuracy_score(yTest, nn.predict(xTest).round())
    # print('Neural Network: trainScore', trainScore)
    # print('Neural Network: testScoret', testScore)
    return trainScore, testScore

def NeuralNetworkLearner_parameter(xTrain, xTest, yTrain, yTest, p):
    hiddenlayer = tuple(100 - 9*i for i in range(1,p))
    nn = MLPClassifier(activation = 'logistic', solver = 'sgd', hidden_layer_sizes=hiddenlayer, learning_rate_init= 0.001).fit(xTrain, yTrain) #training time cooresponding to hidden layer size
    trainScore = accuracy_score(yTrain, nn.predict(xTrain).round())
    testScore = accuracy_score(yTest, nn.predict(xTest).round())
    # print('Neural Network: trainScore', trainScore)
    # print('Neural Network: testScoret', testScore)
    return trainScore, testScore
# ###################################### Neural Network Experiment with 2 dataset & plot ############################################
nn_TrainResult1, nn_TestResult1, nn_TrainResult2, nn_TestResult2 = experiment_size(100, NeuralNetworkLearner)
plotAccuray_TrainSize(nn_TrainResult1, nn_TestResult1,nn_TrainResult2, nn_TestResult2,  "Neural Network", "Training size(1%~99%)")
nn_TrainResult1, nn_TestResult1, nn_TrainResult2, nn_TestResult2 = experiment_parameter(10, NeuralNetworkLearner_parameter)
plotAccuray_parameter(nn_TrainResult1, nn_TestResult1,nn_TrainResult2, nn_TestResult2,  "Neural Network(parameter)","Parameter: hidden layer", 10)



# ###################################### Cross Validation: after determining the best size & parameter ###########################################
def CV_score(clf, clf_name):

    score1 = cross_validate(clf, x_data1, y_data1, cv = 5, scoring='f1_macro')
    score2 = cross_validate(clf, x_data2, y_data2, cv=5, scoring='f1_macro')
    cvScore1 = score1['test_score']
    cvScore2 = score2['test_score']
    time1 = score1['fit_time']
    time2 = score2['fit_time']
    print("Cross Validation Score for ",str(clf_name),"DataSet1: ", str(round(100*cvScore1.mean(), 2)), "%")
    print("Cross Validation Score for ",str(clf_name),"DataSet2: ", str(round(100*cvScore2.mean(), 2)), "%")
    print("FitTime for ",str(clf_name),"DataSet1: ", str(round(100*time1.mean(), 2)))
    print("FitTime for ", str(clf_name), "DataSet2: ", str(round(100*time2.mean(), 2)))
    return

dt = tree.DecisionTreeClassifier(max_depth=6)
CV_score(dt, "DecisionTree")

nn = MLPClassifier()
CV_score(nn, "NeuralNetwork")

boost = AdaBoostClassifier(n_estimators=15)
CV_score(boost, "Boosting")

svm =  SVC(gamma="scale", kernel='rbf')
CV_score(svm, "SVM")

knn = KNeighborsClassifier(n_neighbors=30)
CV_score(knn, "KNN")


