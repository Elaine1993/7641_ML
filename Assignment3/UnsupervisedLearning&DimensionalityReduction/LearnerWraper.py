import numpy as np
import pandas as pd
from sklearn.metrics import silhouette_samples, silhouette_score
import matplotlib.pyplot as plt
import timeit
from sklearn.cluster import KMeans
from sklearn.metrics import silhouette_score
from sklearn.metrics.cluster import normalized_mutual_info_score
from sklearn.preprocessing import MinMaxScaler
from sklearn.mixture import GaussianMixture

from sklearn.decomposition import PCA, FastICA
from sklearn.random_projection import GaussianRandomProjection


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
# print("After-cleaning data1\n", data1.head())


# data resource2: https://archive.ics.uci.edu/ml/datasets/default+of+credit+card+clients
data2 = pd.read_excel("default_of_credit_card_clients.xls", skiprows=[1], index_col= 0)
# print("before-cleaning data2\n", data2.head())
# Separate out the x_data and y_data.
x_data2 = data2.loc[:, data2.columns != "Y"]
y_data2 = data2.loc[:, "Y"]
# print("After-cleaning data2\n", data2.head())

# normalrize Data-------CHECK IF NEED TO NORMALRIZE
scaler1 = MinMaxScaler().fit(x_data1)
x_data1_norm = pd.DataFrame(scaler1.transform(x_data1))

scaler2 = MinMaxScaler().fit(x_data2)
x_data2_norm = pd.DataFrame(scaler2.transform(x_data2))
# print("After-normalize data1\n", x_data1_norm.head())
# print("After-normalize data2\n", x_data2_norm.head())
# ############################################### Helper Functions ###################################################
# experiment_parameter: iteration for n time, each time, change parameter
def experiment_iters(max_iters, f):
    print("TrainResult_accuracy =")
    print("TestResult_accuracy =")
    print("TrainResult_clocktime =")
    print("\n")
    return 0

def plotScore(TrainResult, plotname, y_label, x_label):
    xrange = [2,4,6,8]
    plt.scatter(xrange, TrainResult, label=y_label)
    plt.legend()
    plt.xlabel(x_label)
    plt.ylabel(y_label)
    plt.title(plotname)
    plt.savefig(str(plotname) + ".png")
    plt.show()
    plt.close()

# ###########################################Clustering: K-means ###################################################

def kmean_learner(X, y):
    nclusters = [2,4,6,8] #determine how many cluster:
    sil_scores = [] # track silhouette_score to elvauate cluster density and separation
    traintime = [] # track traning time of each fitting
    nmi_scores = []

    for n in nclusters:
        startTime = timeit.default_timer()
        k_mean = KMeans(n_clusters=n, random_state=10).fit(X)
        totalTime = timeit.default_timer() - startTime
        traintime.append(totalTime)    # calculate time
        ss = silhouette_score(X, k_mean.labels_)
        sil_scores.append(ss)  # calculate sil_score
        nmis = normalized_mutual_info_score(y, k_mean.labels_)
        nmi_scores.append(nmis) # calculate NMI score
        # print("for n=", n, " cluster. The sil_score is: ", ss, ". The nmi_score is: ", nmis)
    print( "time: ", traintime, "The sil_score is: ", sil_scores, ". The nmi_score is: ", nmi_scores)
    return traintime, sil_scores, nmi_scores

traintime, sil_scores, nmi_scores = kmean_learner(x_data1_norm, y_data1) #check normalize data --> n =2 is highest.
#

plotScore(traintime, "Training Time of K-Means", "training time", "Number of Clusters(Dataset1-norm)")
plotScore(sil_scores, "Silhouette Score of K-Means(elbow curve)", "silhouette_score", "Number of Clusters(Dataset1-norm)")
plotScore(nmi_scores, "normalized Mutual Info Score of K-Means", "normalized mutual info score", "Number of Clusters(Dataset1-norm)")
#
traintime, sil_scores, nmi_scores = kmean_learner(x_data1, y_data1) #check normalize data --> n =2 is highest.
#
plotScore(traintime, "Training Time of K-Means", "training time", "Number of Clusters(Dataset1)")
plotScore(sil_scores, "Silhouette Score of K-Means(elbow curve)", "silhouette_score", "Number of Clusters(Dataset1)")
plotScore(nmi_scores, "Normalized Mutual Info Score of K-Means", "normalized mutual info score", "Number of Clusters(Dataset1)")
#
#
#
traintime, sil_scores, nmi_scores = kmean_learner(x_data2, y_data2) #check normalize data --> n =2 is highest.
#
plotScore(traintime, "Training Time of K-Means -- Dataset2", "training time", "Number of Clusters(Dataset2-norm)")
plotScore(sil_scores, "Silhouette Score of K-Means(elbow curve)-- Dataset2", "silhouette_score", "Number of Clusters(Dataset2-norm)")
plotScore(nmi_scores, "Normalized Mutual Info Score of K-Means-- Dataset2", "normalized mutual info score", "Number of Clusters(Dataset2-norm)")

# ##########################################Clustering:Expectation Maximization clustering########################################

def EM_learner(X, y):
    nclusters = [2,4,6,8] #determine how many cluster:
    sil_scores_em = [] # track silhouette_score to elvauate cluster density and separation
    traintime_em = [] # track traning time of each fitting
    nmi_scores_em = []

    for n in nclusters:
        startTime = timeit.default_timer()
        em = GaussianMixture(n_components=n, random_state=10).fit(X)
        totalTime = timeit.default_timer() - startTime
        traintime_em.append(totalTime)    # calculate time
        ss = silhouette_score(X, em.predict(X))
        sil_scores_em.append(ss)  # calculate sil_score
        nmis = normalized_mutual_info_score(y, em.predict(X))
        nmi_scores_em.append(nmis) # calculate NMI score
        # print("for n=", n, " cluster. The sil_score is: ", ss, ". The nmi_score is: ", nmis)
    print( "time: ", traintime_em, "The sil_score is: ", sil_scores_em, ". The nmi_score is: ", nmi_scores_em)
    return traintime_em, sil_scores_em, nmi_scores_em

traintime_em, sil_scores_em, nmi_scores_em = EM_learner(x_data1_norm, y_data1) #check normalize data --> n =2 is highest.
#
plotScore(traintime_em, "Training Time of EM -- Dataset1", "training time", "Number of Clusters(Dataset1-norm)")
plotScore(sil_scores_em, "Silhouette Score of EM -- Dataset1", "training time", "Number of Clusters(Dataset1-norm)")
plotScore(nmi_scores_em, "Normalized Mutual Info Score of K-Means -- Dataset1", "training time", "Number of Clusters(Dataset1-norm)")
#
traintime_em2, sil_scores_em2, nmi_scores_em2 = EM_learner(x_data2_norm, y_data2) #check normalize data --> n =2 is highest.
plotScore(traintime_em2, "Training Time of EM -- Dataset2", "training time", "Number of Clusters(Dataset2-norm)")
plotScore(sil_scores_em2, "Silhouette Score of EM -- Dataset2", "training time", "Number of Clusters(Dataset2-norm)")
plotScore(nmi_scores_em2, "Normalized Mutual Info Score of EM -- Dataset2", "training time", "Number of Clusters(Dataset2-norm)")

 ########################################## Dimensionality Reduction ########################################

def PCA_reduction(X, plt_name):
    pca = PCA(random_state=10).fit(X)
    explained_variance=pca.explained_variance_
    print("explained_variance_ratio", explained_variance)
    plt.figure()
    plt.plot(explained_variance)
    plt.title(plt_name)
    plt.ylabel("explained_variance score")
    plt.xlabel("PCA selected features")
    plt.show()

PCA_reduction(x_data1_norm, "PCA Explained_variance --Dataset1(Normalized)")
PCA_reduction(x_data1, "PCA Explained_variance --Dataset1")
#
PCA_reduction(x_data2_norm, "PCA Explained_Variance --Dataset2(Normalized)")
PCA_reduction(x_data2, "PCA Explained Variance --Dataset2")


def ICA_reduction(X, plt_name, n, ncomponent):
    ncomponent = ncomponent
    ica = FastICA(n_components = n, random_state=10)

    Kurtosis_Across_IC = []
    kaic= ica.fit_transform(X)
    kaic = pd.DataFrame(kaic)
    kaic= kaic.kurt(axis=0)
    Kurtosis_Across_IC.append(kaic)
    print("kaic:", kaic.abs().mean())

    plt.figure()
    plt.scatter(ncomponent, Kurtosis_Across_IC)
    plt.title(plt_name)
    plt.ylabel("Kurtosis Across IC score")
    plt.xlabel("Independent Components")
    plt.show()

ICA_reduction(x_data1_norm, "ICA Kurtosis_Across_IC --Dataset1(Normalized)",4, [1,2,3,4])
ICA_reduction(x_data2_norm, "ICA Kurtosis_Across_IC --Dataset2(Normalized)",5,[1,2,3,4,5])

def RCA_reduction(X, plt_name, n, ncomponent):
    ncomponent = ncomponent
    rca = GaussianRandomProjection(n_components = n, random_state=10)

    pairwiseDistCorr = []
    kaic= rca.fit_transform(X)
    kaic = pd.DataFrame(kaic)
    kaic= kaic.kurt(axis=0)
    pairwiseDistCorr.append(kaic)
    print("kaic:", kaic.abs().mean())

    plt.figure()
    plt.scatter(ncomponent, pairwiseDistCorr)
    plt.title(plt_name)
    plt.ylabel("pairwiseDistCorr")
    plt.xlabel("Random Projection")
    plt.show()

RCA_reduction(x_data1_norm, "RCA --Dataset1(Normalized)",4, [1,2,3,4])
RCA_reduction(x_data2_norm, "RCA --Dataset2(Normalized)",5,[1,2,3,4,5])

def random_forest(X, y):
    rfc = random_forest(random_state =10).fit(X, y)
    rfc.feature_importances_

########################Recreating Clustering Experiment (k-means and EM) for dataset 1 ########################

def recallKMeans_pca(X):
    pca = PCA(4)
    X_new = pca.fit(X).transform(X)
    return X_new
def recallKMeans_ica(X):
    ica=FastICA(4)
    X_new = ica.fit(X).transform(X)
    return X_new
def recallKMeans_rca(X):
    ica=GaussianRandomProjection(4)
    X_new = ica.fit(X).transform(X)
    return X_new
X_new_dataset1 = recallKMeans_pca(x_data1_norm)
traintime, sil_scores, nmi_scores = kmean_learner(X_new_dataset1, y_data1) #check normalize data --> n =2 is highest.
plotScore(traintime, "Training time (recall K-Means)", "training time", "Number of Clusters(Dataset1-norm)")
plotScore(sil_scores, "Silhouette Score (recall K-Means)", "silhouette_score", "Number of Clusters(Dataset1-norm)")
plotScore(nmi_scores, "normalized Mutual Info Score (recall K-Means)", "normalized mutual info score", "Number of Clusters(Dataset1-norm)")
#
X_new_dataset1 = recallKMeans_ica(x_data1_norm)
traintime, sil_scores, nmi_scores = kmean_learner(X_new_dataset1, y_data1) #check normalize data --> n =2 is highest.
plotScore(traintime, "Training time (recall K-Means)", "training time", "Number of Clusters(Dataset1-norm)")
plotScore(sil_scores, "Silhouette Score (recall K-Means)", "silhouette_score", "Number of Clusters(Dataset1-norm)")
plotScore(nmi_scores, "normalized Mutual Info Score (recall K-Means)", "normalized mutual info score", "Number of Clusters(Dataset1-norm)")


X_new_dataset1 = recallKMeans_rca(x_data1_norm)
traintime, sil_scores, nmi_scores = kmean_learner(X_new_dataset1, y_data1) #check normalize data --> n =2 is highest.
plotScore(traintime, "Training time (recall K-Means)", "training time", "Number of Clusters(Dataset1-norm)")
plotScore(sil_scores, "Silhouette Score (recall K-Means)", "silhouette_score", "Number of Clusters(Dataset1-norm)")
plotScore(nmi_scores, "normalized Mutual Info Score (recall K-Means)", "normalized mutual info score", "Number of Clusters(Dataset1-norm)")
