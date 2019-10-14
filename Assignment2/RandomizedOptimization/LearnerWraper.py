import mlrose
import numpy as np
import pandas as pd
from sklearn.model_selection import cross_val_score, cross_validate, train_test_split
from sklearn import tree
from sklearn.metrics import accuracy_score
import matplotlib.pyplot as plt
from datetime import datetime



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


# data resource2: https://archive.ics.uci.edu/ml/datasets/default+of+credit+card+clients
data2 = pd.read_excel("default_of_credit_card_clients.xls", skiprows=[1], index_col= 0)
# print("before-cleaning data2\n", data2.head())
# Separate out the x_data and y_data.
x_data2 = data2.loc[:, data2.columns != "Y"]
y_data2 = data2.loc[:, "Y"]


# ############################################### Helper Functions ###################################################
# Split 70% of the data into training and 1-i% into test sets. The random state to use while splitting the data.
random_state = 100
def fixedsplitData(xdata, ydata):
     xTrain, xTest, yTrain, yTest = train_test_split(xdata, ydata, train_size = 0.7, random_state=random_state)
     return xTrain, xTest, yTrain, yTest

# experiment_parameter: iteration for n time, each time, change parameter
def experiment_iters(max_iters, f):
    xTrain1, xTest1, yTrain1, yTest1= fixedsplitData(x_data1, y_data1)
    TrainResult_accuracy = []
    TestResult_accuracy = []
    TrainResult_time = []
    iter_candidate = list(range(100, max_iters, 200))
    for i in iter_candidate:
        trainScore1, testScore1, TrainResult_clocktime = f(xTrain1, xTest1, yTrain1, yTest1, i)
        TrainResult_accuracy.append(trainScore1)
        TestResult_accuracy.append(testScore1)
        TrainResult_time.append(TrainResult_clocktime)
        #print(TrainResult_clocktime)
    print("TrainResult_accuracy =", TrainResult_accuracy)
    print("TestResult_accuracy =", TestResult_accuracy)
    print("TrainResult_clocktime =", TrainResult_time)
    print("\n")
    return TrainResult_accuracy, TestResult_accuracy, TrainResult_time


def plotAccuray_Accuracy(TrainResult1,TestResult1, plotname, parameter_name):
    xrange = list(range(100, 2000, 200))
    plt.plot(xrange, TrainResult1, label="TrainAccuracy")
    plt.plot(xrange, TestResult1, label="TestAccuracy")
    plt.legend()
    plt.xlabel(parameter_name)
    plt.ylabel("Accuracy")
    plt.title("Accuracy Score of " + str(plotname))
    plt.savefig("AccuracyScore_" + str(plotname) + ".png")
    plt.show()
    plt.close()

def plotAccuray_Time(TrainResult1, plotname, parameter_name):
    xrange = list(range(100, 2001, 200))
    plt.plot(xrange, TrainResult1, label="TrainTime")
    plt.legend()
    plt.xlabel(parameter_name)
    plt.ylabel("Seconds")
    plt.title("Total Training Time of " + str(plotname))
    plt.savefig("TrainingTime_" + str(plotname) + ".png")
    plt.show()
    plt.close()

# ############################################### random hill climbing ###################################################
def RandomHillClimbLearner(xTrain, xTest, yTrain, yTest, max_iters):
    # Create a RandomHillClimb classifier and train it.
    RandomHillClimb = mlrose.NeuralNetwork(hidden_nodes=[20, 20], activation='relu', \
                                          algorithm='random_hill_climb', max_iters=max_iters, \
                                          bias=True, is_classifier=True, learning_rate=1, \
                                          early_stopping=True, clip_max=1e+10, max_attempts=100, \
                                          random_state=random_state, curve=True, restarts=0)
    startTime = datetime.now()
    RandomHillClimb.fit(xTrain, yTrain, init_weights=None)
    totalTime = datetime.now() - startTime
    totalTime = totalTime.total_seconds()
    # Note: Round the output values greater than or equal to 0.5 to 1 and those less than 0.5 to 0. You can use y_predict.round() or any other method.
    trainScore = accuracy_score(yTrain, RandomHillClimb.predict(xTrain).round())
    testScore = accuracy_score(yTest, RandomHillClimb.predict(xTest).round())
    return trainScore, testScore, totalTime

# ############################################### random hill climbing Experiment with 2 data_sets & plot ###################################################
RandomHillClimb_TrainResultAccuracy, RandomHillClimb_TestAccuracy, RandomHillClimb_TrainResultTime = experiment_iters(2001, RandomHillClimbLearner)
plotAccuray_Accuracy(RandomHillClimb_TrainResultAccuracy, RandomHillClimb_TestAccuracy, "Random Hill Climbing(non-restart)", "Max_iteration times")
plotAccuray_Time(RandomHillClimb_TrainResultTime, "Random Hill Climbing(non-restart)", "Max_iteration times")


# ############################################### simulated annealing ###################################################
def simulatedAnnealLearner(xTrain, xTest, yTrain, yTest, max_iters):
    # Create a simulatedAnneal classifier and train it.
    simulatedAnneal = mlrose.NeuralNetwork(hidden_nodes=[20, 20], activation='relu', \
                                          algorithm='simulated_annealing', max_iters=max_iters, \
                                          bias=True, is_classifier=True, learning_rate=1, \
                                          early_stopping=True, clip_max=1e+10, max_attempts=100, \
                                          random_state=random_state, curve=True, schedule=  mlrose.GeomDecay())
    startTime = datetime.now()
    simulatedAnneal.fit(xTrain, yTrain, init_weights=None)
    totalTime = datetime.now() - startTime
    totalTime = totalTime.total_seconds()
    # Note: Round the output values greater than or equal to 0.5 to 1 and those less than 0.5 to 0. You can use y_predict.round() or any other method.
    trainScore = accuracy_score(yTrain, simulatedAnneal.predict(xTrain).round())
    testScore = accuracy_score(yTest, simulatedAnneal.predict(xTest).round())
    return trainScore, testScore, totalTime

# ###############################################simulatedAnneal experiment & plot ###################################################
simulatedAnneal_TrainResultAccuracy, simulatedAnneal_TestAccuracy, simulatedAnneal_TrainResultTime = experiment_iters(2001, simulatedAnnealLearner)
plotAccuray_Accuracy(simulatedAnneal_TrainResultAccuracy, simulatedAnneal_TestAccuracy, "SimulatedAnneal", "Max_iteration times")
plotAccuray_Time(simulatedAnneal_TrainResultTime, "SimulatedAnneal", "Max_iteration times")


# ############################################### genetic algorithm ###################################################
def GeneticAlgorithmLearner(xTrain, xTest, yTrain, yTest, max_iters):
    # Create a GeneticAlgorithm classifier and train it.
    Genetic = mlrose.NeuralNetwork(hidden_nodes=[20, 20], activation='tanh', \
                                          algorithm='genetic_alg', max_iters= max_iters, \
                                          bias=True, is_classifier=True, learning_rate=100, \
                                          early_stopping=True, clip_max=1e+10, max_attempts=100, \
                                           curve=True, pop_size=100, mutation_prob = 0.1)
    startTime = datetime.now()
    Genetic.fit(xTrain, yTrain, init_weights=None)
    totalTime = datetime.now() - startTime
    totalTime = totalTime.total_seconds()
    # Note: Round the output values greater than or equal to 0.5 to 1 and those less than 0.5 to 0. You can use y_predict.round() or any other method.
    trainScore = accuracy_score(yTrain, Genetic.predict(xTrain).round())
    testScore = accuracy_score(yTest, Genetic.predict(xTest).round())
    return trainScore, testScore, totalTime

# ###############################################simulatedAnneal experiment & plot ###################################################
Genetic_TrainResultAccuracy, Genetic_TestAccuracy, Genetic_TrainResultTime = experiment_iters(2001, GeneticAlgorithmLearner)
plotAccuray_Accuracy(Genetic_TrainResultAccuracy, Genetic_TestAccuracy, "GeneticAlgorithm", "Max_iteration times")
plotAccuray_Time(Genetic_TrainResultTime, "GeneticAlgorithm", "Max_iteration times")






