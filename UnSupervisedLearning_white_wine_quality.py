import pandas as pd
import numpy as np
import matplotlib
matplotlib.use('Agg')
import matplotlib.pyplot as plt
import os

from sklearn.model_selection import GridSearchCV

from sklearn.preprocessing import StandardScaler

from sklearn.cluster import KMeans
from sklearn.mixture import GaussianMixture

from sklearn.decomposition import PCA
from sklearn.decomposition import FastICA
from sklearn.random_projection import GaussianRandomProjection,SparseRandomProjection
from sklearn.feature_selection import RFE
from sklearn.svm import SVR

from sklearn.metrics import accuracy_score

from collections import Counter

from collections import defaultdict

from datetime import datetime

# mydir = r"C:\Users\Bryan Marthin\Documents\GT OMS Analytics\Fall 2018\CS 7641 - Machine Learning\Homework\Unsupervised_Learning"
# os.chdir(mydir)
mydir = 'Wine_Graphs/'

np.random.seed(10)

''' function to compute accuracy between 2 set of labels '''
def my_accuracy_cluster(datalabel,resultlabel):

    # create an empty array with shape that match the data label
    myprediction = np.empty_like(datalabel)

    '''get unique value in the result label,
    then based on the position of the points in a cluster we get the most label that occured in that positions,
    that will be our 'guess' for our cluster labels

    since most_common function return a dictionary, to get the most common label, we need to get the key of that dictionary

    '''

    for l in set(resultlabel):
        mask = resultlabel == l
        target = Counter(datalabel[mask]).most_common(1)[0][0]
        myprediction[mask] = target
    return accuracy_score(datalabel,myprediction)

''' fucntion to plot # clusters vs score '''

def plot_score_curve(datadictionary,mytitle) :

    fig=plt.figure()

    num_clusters = list(datadictionary.keys())
    score_clusters = list(datadictionary.values())

    ax = fig.add_subplot(111,xlabel='# Clusters',ylabel='Score',title=mytitle)

    ax.plot(num_clusters, score_clusters, 'o-', color="b",
             label="Num of Clusters")
    ax.set_xticks(num_clusters)

    ax.legend(loc="best")
    fig.savefig(mydir+r"\HW3_"+mytitle+".png")   # save the figure to file
    plt.close(fig)
    return plt

def plot_time_curve(datadictionary,mytitle) :

    fig=plt.figure()

    num_clusters = list(datadictionary.keys())
    time_clusters = list(datadictionary.values())

    ax = fig.add_subplot(111,xlabel='# Clusters',ylabel='Time',title=mytitle)

    ax.plot(num_clusters, time_clusters, 'o-', color="b",
             label="Num of Clusters")
    ax.set_xticks(num_clusters)

    ax.legend(loc="best")
    fig.savefig(mydir+r"\HW3_"+mytitle+".png")   # save the figure to file
    plt.close(fig)
    return plt

def plot_score_feature_transform(datadictionary,mytitle) :

    fig=plt.figure()

    PCA_components = list(datadictionary.keys())

    ax = fig.add_subplot(111,xlabel='# Clusters',ylabel='Score',title=mytitle)

    colors = ['b','g','r','c','m','y','k','w']

    for PCA_comp in range(len(PCA_components)):
        num_clusters = list(datadictionary[PCA_components[PCA_comp]].keys())
        score_clusters = list(datadictionary[PCA_components[PCA_comp]].values())
        ax.plot(num_clusters, score_clusters, 'o-', color=colors[PCA_comp],
                 label=str(PCA_components[PCA_comp]))

    ax.set_xticks(num_clusters)

    ax.legend(loc="best")
    fig.savefig(mydir+r"\HW3_"+mytitle+".png")   # save the figure to file
    plt.close(fig)
    return plt

def plot_time_feature_transform(datadictionary,mytitle) :

    fig=plt.figure()

    PCA_components = list(datadictionary.keys())

    ax = fig.add_subplot(111,xlabel='# Clusters',ylabel='Time',title=mytitle)

    colors = ['b','g','r','c','m','y','k','w']

    for PCA_comp in range(len(PCA_components)):
        num_clusters = list(datadictionary[PCA_components[PCA_comp]].keys())
        time_clusters = list(datadictionary[PCA_components[PCA_comp]].values())
        ax.plot(num_clusters, time_clusters, 'o-', color=colors[PCA_comp],
                 label=str(PCA_components[PCA_comp]))

    ax.set_xticks(num_clusters)

    ax.legend(loc="best")
    fig.savefig(mydir+r"\HW3_"+mytitle+".png")   # save the figure to file
    plt.close(fig)
    return plt

if __name__ == '__main__':

    ''' load wine data '''
    white_wine_data_orig = pd.read_csv("winequality-white.csv",delimiter=";")

    white_wine_data_count = white_wine_data_orig['quality'].value_counts().sort_index()

    white_wine_data_count.index

    ''' plot the data count '''
    white_wine_data_count = white_wine_data_orig[['quality','chlorides']].groupby(['quality'],as_index=False)
    white_wine_data_count = white_wine_data_count.count()
    white_wine_data_count.rename(columns={'chlorides':'count'},inplace=True)

    fig = plt.figure(figsize =(15,10))
    ax = fig.add_subplot(111,xlabel='quality',ylabel='count',title='White Wine Quality Data')
    ax.bar(white_wine_data_count['quality'],white_wine_data_count['count'])
    ax.set_xticks(white_wine_data_count['quality'])
    for item in ([ax.title, ax.xaxis.label, ax.yaxis.label] +
                 ax.get_xticklabels() + ax.get_yticklabels()):
        item.set_fontsize(20)
    fig.savefig(mydir+'\white_wine_grouped_data.png')   # save the figure to file
    plt.close(fig)

    print("There are " + str(white_wine_data_orig.shape[0]) + " rows of data")


    ''' split the data into attributes and labels '''
    ww_orig_attributes = white_wine_data_orig[white_wine_data_orig.columns.difference(['quality'])]
    ww_orig_labels = white_wine_data_orig['quality']

    ''' standardize the attributes '''
    scaler=StandardScaler().fit(ww_orig_attributes)

    ww_orig_attributes[ww_orig_attributes.columns.difference(['quality'])] = scaler.fit_transform(ww_orig_attributes[ww_orig_attributes.columns.difference(['quality'])])

    ''' different clusters '''

    mycluster_ww = [2,3,4,5,6,7,10,15,20,25,30,40,50]



    ''''''''''''''''''''''''''''''''''''''''''''''''''''''''''''''''''''''''''''''
    '''k - means cluster and EM (using gaussian mixture) with no feature selection'''
    ''''''''''''''''''''''''''''''''''''''''''''''''''''''''''''''''''''''''''''''

    my_accuracy_kmeans = dict()
    my_time_kmeans = dict()

    my_accuracy_em = dict()
    my_time_em = dict()

    for myk in mycluster_ww:

        ''' kMeans clustering '''

        startTime = datetime.now()
        myk_mean_prediction = KMeans(n_clusters=myk,random_state=0).fit_predict(ww_orig_attributes)
        myk_mean_accuracy_res = my_accuracy_cluster(ww_orig_labels,myk_mean_prediction)
        endTime = datetime.now()

        # append accuracy
        my_accuracy_kmeans[myk] = myk_mean_accuracy_res
        # append my_time array
        my_time_kmeans[myk] = (endTime-startTime).total_seconds()

        ''' EM using GaussianMixture clustering '''

        startTime = datetime.now()
        my_em_prediction = GaussianMixture(n_components=myk,random_state=0).fit(ww_orig_attributes).predict(ww_orig_attributes)
        my_accuracy_em_res = my_accuracy_cluster(ww_orig_labels,my_em_prediction)
        endTime = datetime.now()

        # append accuracy
        my_accuracy_em[myk] = my_accuracy_em_res
        # append my_time array
        my_time_em[myk] = (endTime-startTime).total_seconds()

    plot_score_curve(my_accuracy_kmeans,"k-means clusters vs score")
    plot_time_curve(my_time_kmeans,"k-means clusters vs time")

    plot_score_curve(my_accuracy_em,"EM clusters vs score")
    plot_time_curve(my_time_em,"EM clusters vs time")

    ''''''''''''''''''''''''''''''''''''''''''''''''''''''''''''''''''''''''''''''
    '''k - means cluster and EM (using gaussian mixture) after PCA'''
    ''''''''''''''''''''''''''''''''''''''''''''''''''''''''''''''''''''''''''''''

    # for whitewine PCA, we can only have 11 Principal components since the number of features for white wine is 11
    PCA_component_whitewine = [1,2,3,5,8,11]

    my_accuracy_kmeans_PCA = defaultdict(dict)
    my_time_kmeans_PCA = defaultdict(dict)
    my_accuracy_em_PCA = defaultdict(dict)
    my_time_em_PCA = defaultdict(dict)

    white_wine_data_PCA = PCA(random_state=0)
    ww_data_eigen  = white_wine_data_PCA.fit(ww_orig_attributes)
    ww_data_eigenvalues = ww_data_eigen.explained_variance_

    for PCA_comp in PCA_component_whitewine :

        white_wine_data_PCA = PCA(n_components=PCA_comp,random_state=0)
        white_wine_data_PCA_data = white_wine_data_PCA.fit_transform(ww_orig_attributes)
        white_wine_data_PCA_df = pd.DataFrame(data = white_wine_data_PCA_data)

        white_wine_data_PCA_df_nn = pd.concat([white_wine_data_PCA_df,ww_orig_labels],axis=1)

        for cluster in mycluster_ww:

            ''' kMeans clustering '''
            startTime = datetime.now()
            myk_mean_PCA_prediction = KMeans(n_clusters=cluster,random_state=0).fit_predict(white_wine_data_PCA_df)
            myk_mean_accuracy_res = my_accuracy_cluster(ww_orig_labels,myk_mean_PCA_prediction)
            endTime = datetime.now()
            # append accuracy
            my_accuracy_kmeans_PCA[PCA_comp][cluster] = myk_mean_accuracy_res
            # append my_time array
            my_time_kmeans_PCA[PCA_comp][cluster] = (endTime-startTime).total_seconds()

            ''' EM using GaussianMixture clustering '''
            startTime = datetime.now()
            my_em_prediction = GaussianMixture(n_components=cluster).fit(white_wine_data_PCA_df).predict(white_wine_data_PCA_df)
            my_accuracy_em_res = my_accuracy_cluster(ww_orig_labels,my_em_prediction)
            endTime = datetime.now()

            # append accuracy
            my_accuracy_em_PCA[PCA_comp][cluster] = my_accuracy_em_res
            # append my_time array
            my_time_em_PCA[PCA_comp][cluster] = (endTime-startTime).total_seconds()

    plot_time_feature_transform(my_time_kmeans_PCA,"k-means PCA clusters vs time")
    plot_score_feature_transform(my_accuracy_kmeans_PCA,"k-means PCA clusters vs score")
    plot_time_feature_transform(my_time_em_PCA,"EM PCA clusters vs time")
    plot_score_feature_transform(my_accuracy_em_PCA,"EM PCA clusters vs score")

    ''' to illustrate the data in 2 component PCA '''
    white_wine_data_PCA = PCA(n_components=2,random_state=0)
    white_wine_data_PCA_data = white_wine_data_PCA.fit_transform(ww_orig_attributes)
    white_wine_data_PCA_df = pd.DataFrame(data = white_wine_data_PCA_data)

    fig = plt.figure(figsize = (8,8))
    ax = fig.add_subplot(1,1,1)
    ax.set_xlabel('Principal Component 1', fontsize = 15)
    ax.set_ylabel('Principal Component 2', fontsize = 15)
    ax.set_title('2 component PCA', fontsize = 20)
    targets = list(set(ww_orig_labels))
    colors = ['r', 'g', 'b','c','m','y','k']
    for target, color in zip(targets,colors):
        indicesToKeep = ww_orig_labels == target
        ax.scatter(white_wine_data_PCA_df.loc[indicesToKeep,0]
                   , white_wine_data_PCA_df.loc[indicesToKeep,1]
                   , c = color
                   , s = 50)
    ax.legend(targets)
    ax.grid()
    fig.savefig(mydir+r"\HW3_"+"data_in_2_PCA"+".png")   # save the figure to file

    ''''''''''''''''''''''''''''''''''''''''''''''''''''''''''''''''''''''''''''''
    '''k - means cluster and EM (using gaussian mixture) after ICA'''
    ''''''''''''''''''''''''''''''''''''''''''''''''''''''''''''''''''''''''''''''
    # for whitewine ICA, we can only have 11 Principal components since the number of features for white wine is 11
    ICA_component_whitewine = [1,2,4,6,8,11]

    my_accuracy_kmeans_ICA = defaultdict(dict)
    my_time_kmeans_ICA = defaultdict(dict)
    my_accuracy_em_ICA = defaultdict(dict)
    my_time_em_ICA = defaultdict(dict)

    white_wine_data_ICA = FastICA(random_state=0)
    white_wine_data_ICA_data = white_wine_data_ICA.fit_transform(ww_orig_attributes)
    white_wine_data_ICA_df = pd.DataFrame(data = white_wine_data_ICA_data)
    white_wine_data_ICA_kurtosis = white_wine_data_ICA_df.kurt()

    for ICA_comp in ICA_component_whitewine :

        white_wine_data_ICA = FastICA(n_components=ICA_comp,random_state=0)
        white_wine_data_ICA_data = white_wine_data_ICA.fit_transform(ww_orig_attributes)
        white_wine_data_ICA_df = pd.DataFrame(data = white_wine_data_ICA_data)

        for cluster in mycluster_ww:

            ''' kMeans clustering '''
            startTime = datetime.now()
            myk_mean_ICA_prediction = KMeans(n_clusters=cluster,random_state=0).fit_predict(white_wine_data_ICA_df)
            myk_mean_accuracy_res = my_accuracy_cluster(ww_orig_labels,myk_mean_ICA_prediction)
            endTime = datetime.now()
            # append accuracy
            my_accuracy_kmeans_ICA[ICA_comp][cluster] = myk_mean_accuracy_res
            # append my_time array
            my_time_kmeans_ICA[ICA_comp][cluster] = (endTime-startTime).total_seconds()

            ''' EM using GaussianMixture clustering '''
            startTime = datetime.now()
            my_em_prediction = GaussianMixture(n_components=cluster).fit(white_wine_data_ICA_df).predict(white_wine_data_ICA_df)
            my_accuracy_em_res = my_accuracy_cluster(ww_orig_labels,my_em_prediction)
            endTime = datetime.now()

            # append accuracy
            my_accuracy_em_ICA[ICA_comp][cluster] = my_accuracy_em_res
            # append my_time array
            my_time_em_ICA[ICA_comp][cluster] = (endTime-startTime).total_seconds()

    plot_time_feature_transform(my_time_kmeans_ICA,"k-means ICA clusters vs time")
    plot_score_feature_transform(my_accuracy_kmeans_ICA,"k-means ICA clusters vs score")
    plot_time_feature_transform(my_time_em_ICA,"EM ICA clusters vs time")
    plot_score_feature_transform(my_accuracy_em_ICA,"EM ICA clusters vs score")

    ''' to illustrate the data in 2 component ICA '''
    white_wine_data_ICA = FastICA(n_components=2)
    white_wine_data_ICA_data = white_wine_data_ICA.fit_transform(ww_orig_attributes)
    white_wine_data_ICA_df = pd.DataFrame(data = white_wine_data_ICA_data)

    fig = plt.figure(figsize = (8,8))
    ax = fig.add_subplot(1,1,1)
    ax.set_xlabel('Component 1', fontsize = 15)
    ax.set_ylabel('Component 2', fontsize = 15)
    ax.set_title('2 component ICA', fontsize = 20)
    targets = list(set(ww_orig_labels))
    colors = ['r', 'g', 'b','c','m','y','k']
    for target, color in zip(targets,colors):
        indicesToKeep = ww_orig_labels == target
        ax.scatter(white_wine_data_ICA_df.loc[indicesToKeep,0]
                   , white_wine_data_ICA_df.loc[indicesToKeep,1]
                   , c = color
                   , s = 50)
    ax.legend(targets)
    ax.grid()
    fig.savefig(mydir+r"\HW3_"+"data_in_2_ICA"+".png")   # save the figure to file

    ''''''''''''''''''''''''''''''''''''''''''''''''''''''''''''''''''''''''''''''
    '''k - means cluster and EM (using gaussian mixture) after RP'''
    ''''''''''''''''''''''''''''''''''''''''''''''''''''''''''''''''''''''''''''''
    # for whitewine ICA, we can only have 11 Principal components since the number of features for white wine is 11

    RP_component_whitewine = [1,2,4,6,8,11]

    my_accuracy_kmeans_RP = defaultdict(dict)
    my_time_kmeans_RP = defaultdict(dict)
    my_accuracy_em_RP = defaultdict(dict)
    my_time_em_RP = defaultdict(dict)

    white_wine_data_RP = GaussianRandomProjection(random_state=0,n_components=11)
    white_wine_data_RP_data = white_wine_data_RP.fit_transform(ww_orig_attributes)
    white_wine_data_RP_df = pd.DataFrame(data = white_wine_data_RP_data)
    white_wine_data_RP_kurtosis = white_wine_data_RP_df.kurt()

    for RP_comp in RP_component_whitewine :

        white_wine_data_RP = GaussianRandomProjection(n_components=RP_comp,random_state=0)
        white_wine_data_RP_data = white_wine_data_RP.fit_transform(ww_orig_attributes)
        white_wine_data_RP_df = pd.DataFrame(data = white_wine_data_RP_data)

        for cluster in mycluster_ww:

            ''' kMeans clustering '''
            startTime = datetime.now()
            myk_mean_RP_prediction = KMeans(n_clusters=cluster,random_state=0).fit_predict(white_wine_data_RP_df)
            myk_mean_accuracy_res = my_accuracy_cluster(ww_orig_labels,myk_mean_RP_prediction)
            endTime = datetime.now()
            # append accuracy
            my_accuracy_kmeans_RP[RP_comp][cluster] = myk_mean_accuracy_res
            # append my_time array
            my_time_kmeans_RP[RP_comp][cluster] = (endTime-startTime).total_seconds()

            ''' EM using GaussianMixture clustering '''
            startTime = datetime.now()
            my_em_prediction = GaussianMixture(n_components=cluster).fit(white_wine_data_RP_df).predict(white_wine_data_RP_df)
            my_accuracy_em_res = my_accuracy_cluster(ww_orig_labels,my_em_prediction)
            endTime = datetime.now()

            # append accuracy
            my_accuracy_em_RP[RP_comp][cluster] = my_accuracy_em_res
            # append my_time array
            my_time_em_RP[RP_comp][cluster] = (endTime-startTime).total_seconds()

    plot_time_feature_transform(my_time_kmeans_RP,"k-means RP clusters vs time")
    plot_score_feature_transform(my_accuracy_kmeans_RP,"k-means RP clusters vs score")
    plot_time_feature_transform(my_time_em_RP,"EM RP clusters vs time")
    plot_score_feature_transform(my_accuracy_em_RP,"EM RP clusters vs score")

    ''' to illustrate the data in 2 component RP '''
    white_wine_data_RP = GaussianRandomProjection(n_components=2)
    white_wine_data_RP_data = white_wine_data_RP.fit_transform(ww_orig_attributes)
    white_wine_data_RP_df = pd.DataFrame(data = white_wine_data_RP_data)

    fig = plt.figure(figsize = (8,8))
    ax = fig.add_subplot(1,1,1)
    ax.set_xlabel('Component 1', fontsize = 15)
    ax.set_ylabel('Component 2', fontsize = 15)
    ax.set_title('2 component RP', fontsize = 20)
    targets = list(set(ww_orig_labels))
    colors = ['r', 'g', 'b','c','m','y','k']
    for target, color in zip(targets,colors):
        indicesToKeep = ww_orig_labels == target
        ax.scatter(white_wine_data_RP_df.loc[indicesToKeep,0]
                   , white_wine_data_RP_df.loc[indicesToKeep,1]
                   , c = color
                   , s = 50)
    ax.legend(targets)
    ax.grid()
    fig.savefig(mydir+r"\HW3_"+"data_in_2_RP"+".png")   # save the figure to file


    ''''''''''''''''''''''''''''''''''''''''''''''''''''''''''''''''''''''''''''''
    '''k - means cluster and EM (using gaussian mixture) after RFE'''
    ''''''''''''''''''''''''''''''''''''''''''''''''''''''''''''''''''''''''''''''
    # for whitewine ICA, we can only have 11 Principal components since the number of features for white wine is 11
    RFE_component_whitewine = [1,2,4,6,8,11]

    estimator = SVR(kernel="linear")

    my_accuracy_kmeans_RFE = defaultdict(dict)
    my_time_kmeans_RFE = defaultdict(dict)
    my_accuracy_em_RFE = defaultdict(dict)
    my_time_em_RFE = defaultdict(dict)

    for RFE_comp in RFE_component_whitewine :

        white_wine_data_RFE = RFE(estimator,n_features_to_select=RFE_comp)
        white_wine_data_RFE_data = white_wine_data_RFE.fit_transform(ww_orig_attributes,ww_orig_labels)
        white_wine_data_RFE_df = pd.DataFrame(data = white_wine_data_RFE_data)

        for cluster in mycluster_ww:

            ''' kMeans clustering '''
            startTime = datetime.now()
            myk_mean_RFE_prediction = KMeans(n_clusters=cluster,random_state=0).fit_predict(white_wine_data_RFE_df)
            myk_mean_accuracy_res = my_accuracy_cluster(ww_orig_labels,myk_mean_RFE_prediction)
            endTime = datetime.now()
            # append accuracy
            my_accuracy_kmeans_RFE[RFE_comp][cluster] = myk_mean_accuracy_res
            # append my_time array
            my_time_kmeans_RFE[RFE_comp][cluster] = (endTime-startTime).total_seconds()

            ''' EM using GaussianMixture clustering '''
            startTime = datetime.now()
            my_em_prediction = GaussianMixture(n_components=cluster).fit(white_wine_data_RFE_df).predict(white_wine_data_RFE_df)
            my_accuracy_em_res = my_accuracy_cluster(ww_orig_labels,my_em_prediction)
            endTime = datetime.now()

            # append accuracy
            my_accuracy_em_RFE[RFE_comp][cluster] = my_accuracy_em_res
            # append my_time array
            my_time_em_RFE[RFE_comp][cluster] = (endTime-startTime).total_seconds()

    plot_time_feature_transform(my_time_kmeans_RFE,"k-means RFE clusters vs time")
    plot_score_feature_transform(my_accuracy_kmeans_RFE,"k-means RFE clusters vs score")
    plot_time_feature_transform(my_time_em_RFE,"EM RFE clusters vs time")
    plot_score_feature_transform(my_accuracy_em_RFE,"EM RFE clusters vs score")

    ''' to illustrate the data in 2 component RFE '''
    white_wine_data_RFE = RFE(estimator,n_features_to_select=2)
    white_wine_data_RFE_data = white_wine_data_RFE.fit_transform(ww_orig_attributes,ww_orig_labels)
    white_wine_data_RFE_df = pd.DataFrame(data = white_wine_data_RFE_data)

    fig = plt.figure(figsize = (8,8))
    ax = fig.add_subplot(1,1,1)
    ax.set_xlabel('Component 1', fontsize = 15)
    ax.set_ylabel('Component 2', fontsize = 15)
    ax.set_title('2 component RFE', fontsize = 20)
    targets = list(set(ww_orig_labels))
    colors = ['r', 'g', 'b','c','m','y','k']
    for target, color in zip(targets,colors):
        indicesToKeep = ww_orig_labels == target
        ax.scatter(white_wine_data_RFE_df.loc[indicesToKeep,0]
                   , white_wine_data_RFE_df.loc[indicesToKeep,1]
                   , c = color
                   , s = 50)
    ax.legend(targets)
    ax.grid()
    fig.savefig(mydir+r"\HW3_"+"data_in_2_RFE"+".png")   # save the figure to file
