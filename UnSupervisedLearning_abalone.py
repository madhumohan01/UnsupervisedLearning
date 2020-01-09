import pandas as pd
import numpy as np
import matplotlib
matplotlib.use('Agg')
import matplotlib.pyplot as plt
import os

from sklearn.model_selection import GridSearchCV

from sklearn.preprocessing import StandardScaler

from sklearn.model_selection import train_test_split

from sklearn.cluster import KMeans
from sklearn.mixture import GaussianMixture

from sklearn.decomposition import PCA
from sklearn.decomposition import FastICA
from sklearn.random_projection import GaussianRandomProjection,SparseRandomProjection

from sklearn.neural_network import MLPClassifier
from sklearn.feature_selection import RFE
from sklearn.svm import SVR

from sklearn.metrics import accuracy_score

from collections import Counter

from collections import defaultdict

from datetime import datetime

mydir = 'Abalone_Graphs/'

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


def NN_learner(data) :
    ''' use Grid Search for cross validating training data set '''
    startTime = datetime.now()
    mydict = dict()

    ''' Split data into attributes and output, then standardize the attributes '''
    abalone_train,abalone_test = train_test_split(data,test_size=0.3,random_state=10)

    aba_train_attributes=abalone_train[abalone_train.columns.difference(['RingsGrouped','Rings'])]
    aba_train_values=abalone_train['RingsGrouped']

    abalone_test_attributes = abalone_test[abalone_train.columns.difference(['RingsGrouped','Rings'])]
    aba_test_values=abalone_test['RingsGrouped']

    param_NN_grid = [{'solver': ['sgd'],'hidden_layer_sizes':[(50,),(100,),(200,)],'max_iter':[50,100],'learning_rate' : ['constant','adaptive'], 'alpha': [0.001], 'learning_rate_init':[0.01]}]

    NN_grid = MLPClassifier(random_state=10)

    NN_score = ['accuracy','f1_micro'] # our score will use
    NN_abalone = GridSearchCV(NN_grid,param_NN_grid,cv=10,refit='f1_micro',scoring=NN_score,return_train_score=True)

    NN_abalone_result = NN_abalone.fit(aba_train_attributes,aba_train_values)

    # get the index result from the grid search that has the highest/best f1_micro score
    best_f1_micro_index= np.where(NN_abalone_result.cv_results_['rank_test_f1_micro']==1)
    toprankindex = best_f1_micro_index[0][0]

    print("best f1 micro score from NN Grid CV is " +str(NN_abalone_result.cv_results_['mean_test_f1_micro'][toprankindex]))

    # get best parameter in terms of solver, alpha, learning rate init and hidden layer size
    best_param_alpha = NN_abalone_result.cv_results_['param_alpha'][toprankindex]
    best_param_solver = NN_abalone_result.cv_results_['param_solver'][toprankindex]
    best_param_learning_rate = NN_abalone_result.cv_results_['param_learning_rate'][toprankindex]
    best_param_learning_rate_init = NN_abalone_result.cv_results_['param_learning_rate_init'][toprankindex]
    best_param_hidden_layer_sizes = NN_abalone_result.cv_results_['param_hidden_layer_sizes'][toprankindex]


    ''' create the best decision tree model with best paramater using lbfgs'''

    if best_param_solver =='sgd' :
        Best_NN_from_cv = MLPClassifier(random_state=10,
                                        solver=best_param_solver,
                                        alpha=best_param_alpha,
                                        learning_rate=best_param_learning_rate,learning_rate_init=best_param_learning_rate_init,
                                        hidden_layer_sizes=best_param_hidden_layer_sizes)
    elif best_param_solver == 'adam':
            Best_NN_from_cv = MLPClassifier(random_state=10,
                                        solver=best_param_solver,
                                        alpha=best_param_alpha,
                                        learning_rate_init=best_param_learning_rate_init)
    else :
            Best_NN_from_cv = MLPClassifier(random_state=10,
                                        solver=best_param_solver,
                                        alpha=best_param_alpha)

    aba_bestNN = Best_NN_from_cv.fit(aba_train_attributes,aba_train_values) #toremove
    aba_bestNN.predict(abalone_test_attributes) #toremove
    aba_bestNN.score(abalone_test_attributes,aba_test_values) #toremove
    endtime = datetime.now()
    mytime=(endtime-startTime).total_seconds()
    bestNNparam = [best_param_solver,best_param_alpha,best_param_learning_rate,best_param_learning_rate_init,best_param_hidden_layer_sizes]


    return [aba_bestNN.score(abalone_test_attributes,aba_test_values),mytime,bestNNparam]

if __name__ == '__main__':

    ''' load abalone data '''
    # abalone data
    abalone_data = pd.read_csv(r"abalone.data.csv",delimiter=",")
    abalone_data['Sex']=pd.factorize(abalone_data['Sex'])[0]#change sex column into categorical

    abalone_data_count = abalone_data['Rings'].value_counts().sort_index()

    abalone_data_count.plot('bar',figsize=(15,10))

    abalone_data.index

    abalone_data['RingsGrouped'] = np.where(abalone_data['Rings']<=8, 0,
                                   np.where(abalone_data['Rings']>=11, 2,
                                   np.where((abalone_data['Rings']==9) | (abalone_data['Rings']==10), 1,0)))


    ''' plot the data count '''
    abalone_data_count = abalone_data[['RingsGrouped','Length']].groupby(['RingsGrouped'],as_index=False)
    abalone_data_count = abalone_data_count.count()
    abalone_data_count.rename(columns={'Length':'count'},inplace=True)

    fig = plt.figure(figsize =(15,10))
    ax = fig.add_subplot(111,xlabel='Rings',ylabel='count',title='Abalone Ring Data')
    ax.bar(abalone_data_count['RingsGrouped'],abalone_data_count['count'])

    ax.set_xticks(abalone_data_count['RingsGrouped'])
    ax.set_xticklabels(['Rings less than 9','Rings between 9 and 10','Rings more than 10'])
    for item in ([ax.title, ax.xaxis.label, ax.yaxis.label] +
                 ax.get_xticklabels() + ax.get_yticklabels()):
        item.set_fontsize(20)
    fig.savefig(mydir+r'\abalone_data_count.png')   # save the figure to file
    plt.close(fig)

    print("There are " + str(abalone_data.shape[0]) + " rows of data")

    ''' split the data into attributes and labels '''
    abalone_orig_attributes = abalone_data[abalone_data.columns.difference(['RingsGrouped','Rings'])]
    abalone_orig_labels = abalone_data['RingsGrouped']

    ''' standardize the attributes '''
    scaler=StandardScaler().fit(abalone_orig_attributes)

    abalone_orig_attributes[abalone_orig_attributes.columns.difference(['RingsGrouped','Rings'])] = scaler.fit_transform(abalone_data[abalone_data.columns.difference(['RingsGrouped','Rings'])])

    ''' different clusters '''

    mycluster_aba = [2,3,4,5,6,7,10,15,20,25,30,40,50]



    ''''''''''''''''''''''''''''''''''''''''''''''''''''''''''''''''''''''''''''''
    '''k - means cluster and EM (using gaussian mixture) with no feature selection'''
    ''''''''''''''''''''''''''''''''''''''''''''''''''''''''''''''''''''''''''''''

    my_accuracy_kmeans = dict()
    my_time_kmeans = dict()

    my_accuracy_em = dict()
    my_time_em = dict()

    for myk in mycluster_aba:

        ''' kMeans clustering '''

        startTime = datetime.now()
        myk_mean_prediction = KMeans(n_clusters=myk,random_state=0).fit_predict(abalone_orig_attributes)
        myk_mean_accuracy_res = my_accuracy_cluster(abalone_orig_labels,myk_mean_prediction)
        endTime = datetime.now()

        # append accuracy
        my_accuracy_kmeans[myk] = myk_mean_accuracy_res
        # append my_time array
        my_time_kmeans[myk] = (endTime-startTime).total_seconds()

        ''' EM using GaussianMixture clustering '''

        startTime = datetime.now()
        my_em_prediction = GaussianMixture(n_components=myk,random_state=0).fit(abalone_orig_attributes).predict(abalone_orig_attributes)
        my_accuracy_em_res = my_accuracy_cluster(abalone_orig_labels,my_em_prediction)
        endTime = datetime.now()

        # append accuracy
        my_accuracy_em[myk] = my_accuracy_em_res
        # append my_time array
        my_time_em[myk] = (endTime-startTime).total_seconds()

    abalone_data_orig_df_nn = pd.concat([abalone_orig_attributes,abalone_orig_labels],axis=1)

    NN_ori_score = NN_learner(abalone_data_orig_df_nn)

    plot_score_curve(my_accuracy_kmeans,"k-means clusters vs score")
    plot_time_curve(my_time_kmeans,"k-means clusters vs time")

    plot_score_curve(my_accuracy_em,"EM clusters vs score")
    plot_time_curve(my_time_em,"EM clusters vs time")

    ''''''''''''''''''''''''''''''''''''''''''''''''''''''''''''''''''''''''''''''
    '''k - means cluster and EM (using gaussian mixture) after PCA'''
    ''''''''''''''''''''''''''''''''''''''''''''''''''''''''''''''''''''''''''''''
    ''' dictionary for the neural network '''
    NN_PCA_score = defaultdict(dict)

    # for abalone PCA, we can only have 11 Principal components since the number of features for abalone is 11
    PCA_component_abalone = [1,2,3,4,6,8]

    my_accuracy_kmeans_PCA = defaultdict(dict)
    my_time_kmeans_PCA = defaultdict(dict)
    my_accuracy_em_PCA = defaultdict(dict)
    my_time_em_PCA = defaultdict(dict)

    abalone_data_PCA = PCA(random_state=0)
    aba_data_eigen  = abalone_data_PCA.fit(abalone_orig_attributes)
    aba_data_eigenvalues = aba_data_eigen.explained_variance_

    for PCA_comp in PCA_component_abalone :

        abalone_data_PCA = PCA(n_components=PCA_comp,random_state=0)
        abalone_data_PCA_data = abalone_data_PCA.fit_transform(abalone_orig_attributes)
        abalone_data_PCA_df = pd.DataFrame(data = abalone_data_PCA_data)

        abalone_data_PCA_df_nn = pd.concat([abalone_data_PCA_df,abalone_orig_labels],axis=1)


        for cluster in mycluster_aba:

            ''' kMeans clustering '''
            startTime = datetime.now()
            myk_mean_PCA_prediction = KMeans(n_clusters=cluster,random_state=0).fit_predict(abalone_data_PCA_df)
            myk_mean_accuracy_res = my_accuracy_cluster(abalone_orig_labels,myk_mean_PCA_prediction)
            endTime = datetime.now()
            # append accuracy
            my_accuracy_kmeans_PCA[PCA_comp][cluster] = myk_mean_accuracy_res
            # append my_time array
            my_time_kmeans_PCA[PCA_comp][cluster] = (endTime-startTime).total_seconds()

            ''' EM using GaussianMixture clustering '''
            startTime = datetime.now()
            my_em_prediction = GaussianMixture(n_components=cluster).fit(abalone_data_PCA_df).predict(abalone_data_PCA_df)
            my_accuracy_em_res = my_accuracy_cluster(abalone_orig_labels,my_em_prediction)
            endTime = datetime.now()

            # append accuracy
            my_accuracy_em_PCA[PCA_comp][cluster] = my_accuracy_em_res
            # append my_time array
            my_time_em_PCA[PCA_comp][cluster] = (endTime-startTime).total_seconds()

        ''' NN GridSearch '''
        NN_PCA_score[PCA_comp] = NN_learner(abalone_data_PCA_df_nn)

    plot_time_feature_transform(my_time_kmeans_PCA,"k-means PCA clusters vs time")
    plot_score_feature_transform(my_accuracy_kmeans_PCA,"k-means PCA clusters vs score")
    plot_time_feature_transform(my_time_em_PCA,"EM PCA clusters vs time")
    plot_score_feature_transform(my_accuracy_em_PCA,"EM PCA clusters vs score")

    ''' to illustrate the data in 2 component PCA '''
    abalone_data_PCA = PCA(n_components=2,random_state=0)
    abalone_data_PCA_data = abalone_data_PCA.fit_transform(abalone_orig_attributes)
    abalone_data_PCA_df = pd.DataFrame(data = abalone_data_PCA_data)

    fig = plt.figure(figsize = (8,8))
    ax = fig.add_subplot(1,1,1)
    ax.set_xlabel('Principal Component 1', fontsize = 15)
    ax.set_ylabel('Principal Component 2', fontsize = 15)
    ax.set_title('2 component PCA', fontsize = 20)
    targets = list(set(abalone_orig_labels))
    colors = ['r', 'g', 'b','c','m','y','k']
    for target, color in zip(targets,colors):
        indicesToKeep = abalone_orig_labels == target
        ax.scatter(abalone_data_PCA_df.loc[indicesToKeep,0]
                   , abalone_data_PCA_df.loc[indicesToKeep,1]
                   , c = color
                   , s = 50)
    ax.legend(targets)
    ax.grid()
    fig.savefig(mydir+r"\HW3_"+"data_in_2_PCA"+".png")   # save the figure to file

    ''''''''''''''''''''''''''''''''''''''''''''''''''''''''''''''''''''''''''''''
    '''k - means cluster and EM (using gaussian mixture) after ICA'''
    ''''''''''''''''''''''''''''''''''''''''''''''''''''''''''''''''''''''''''''''
    # for abalone ICA, we can only have 11 Principal components since the number of features for abalone is 11
    ICA_component_abalone = [1,2,3,4,6,8]

    NN_ICA_score = defaultdict(dict)

    my_accuracy_kmeans_ICA = defaultdict(dict)
    my_time_kmeans_ICA = defaultdict(dict)
    my_accuracy_em_ICA = defaultdict(dict)
    my_time_em_ICA = defaultdict(dict)

    abalone_data_ICA = FastICA(random_state=0)
    abalone_data_ICA_data = abalone_data_ICA.fit_transform(abalone_orig_attributes)
    abalone_data_ICA_df = pd.DataFrame(data = abalone_data_ICA_data)
    abalone_data_ICA_kurtosis = abalone_data_ICA_df.kurt()

    for ICA_comp in ICA_component_abalone :

        abalone_data_ICA = FastICA(n_components=ICA_comp,random_state=0)
        abalone_data_ICA_data = abalone_data_ICA.fit_transform(abalone_orig_attributes)
        abalone_data_ICA_df = pd.DataFrame(data = abalone_data_ICA_data)

        abalone_data_ICA_df_nn = pd.concat([abalone_data_ICA_df,abalone_orig_labels],axis=1)
        for cluster in mycluster_aba:

            ''' kMeans clustering '''
            startTime = datetime.now()
            myk_mean_ICA_prediction = KMeans(n_clusters=cluster,random_state=0).fit_predict(abalone_data_ICA_df)
            myk_mean_accuracy_res = my_accuracy_cluster(abalone_orig_labels,myk_mean_ICA_prediction)
            endTime = datetime.now()
            # append accuracy
            my_accuracy_kmeans_ICA[ICA_comp][cluster] = myk_mean_accuracy_res
            # append my_time array
            my_time_kmeans_ICA[ICA_comp][cluster] = (endTime-startTime).total_seconds()

            ''' EM using GaussianMixture clustering '''
            startTime = datetime.now()
            my_em_prediction = GaussianMixture(n_components=cluster).fit(abalone_data_ICA_df).predict(abalone_data_ICA_df)
            my_accuracy_em_res = my_accuracy_cluster(abalone_orig_labels,my_em_prediction)
            endTime = datetime.now()

            # append accuracy
            my_accuracy_em_ICA[ICA_comp][cluster] = my_accuracy_em_res
            # append my_time array
            my_time_em_ICA[ICA_comp][cluster] = (endTime-startTime).total_seconds()

        ''' NN GridSearch '''
        NN_ICA_score[ICA_comp] = NN_learner(abalone_data_ICA_df_nn)

    plot_time_feature_transform(my_time_kmeans_ICA,"k-means ICA clusters vs time")
    plot_score_feature_transform(my_accuracy_kmeans_ICA,"k-means ICA clusters vs score")
    plot_time_feature_transform(my_time_em_ICA,"EM ICA clusters vs time")
    plot_score_feature_transform(my_accuracy_em_ICA,"EM ICA clusters vs score")

    ''' to illustrate the data in 2 component ICA '''
    abalone_data_ICA = FastICA(n_components=2)
    abalone_data_ICA_data = abalone_data_ICA.fit_transform(abalone_orig_attributes)
    abalone_data_ICA_df = pd.DataFrame(data = abalone_data_ICA_data)

    fig = plt.figure(figsize = (8,8))
    ax = fig.add_subplot(1,1,1)
    ax.set_xlabel('Component 1', fontsize = 15)
    ax.set_ylabel('Component 2', fontsize = 15)
    ax.set_title('2 component ICA', fontsize = 20)
    targets = list(set(abalone_orig_labels))
    colors = ['r', 'g', 'b','c','m','y','k']
    for target, color in zip(targets,colors):
        indicesToKeep = abalone_orig_labels == target
        ax.scatter(abalone_data_ICA_df.loc[indicesToKeep,0]
                   , abalone_data_ICA_df.loc[indicesToKeep,1]
                   , c = color
                   , s = 50)
    ax.legend(targets)
    ax.grid()
    fig.savefig(mydir+r"\HW3_"+"data_in_2_ICA"+".png")   # save the figure to file

    ''''''''''''''''''''''''''''''''''''''''''''''''''''''''''''''''''''''''''''''
    '''k - means cluster and EM (using gaussian mixture) after RP'''
    ''''''''''''''''''''''''''''''''''''''''''''''''''''''''''''''''''''''''''''''
    # for abalone ICA, we can only have 11 Principal components since the number of features for abalone is 11

    RP_component_abalone = [1,2,3,4,6,8]

    NN_RP_score = defaultdict(dict)

    my_accuracy_kmeans_RP = defaultdict(dict)
    my_time_kmeans_RP = defaultdict(dict)
    my_accuracy_em_RP = defaultdict(dict)
    my_time_em_RP = defaultdict(dict)

    abalone_data_RP = GaussianRandomProjection(random_state=0,n_components=11)
    abalone_data_RP_data = abalone_data_RP.fit_transform(abalone_orig_attributes)
    abalone_data_RP_df = pd.DataFrame(data = abalone_data_RP_data)
    abalone_data_RP_kurtosis = abalone_data_RP_df.kurt()

    for RP_comp in RP_component_abalone :

        abalone_data_RP = GaussianRandomProjection(n_components=RP_comp,random_state=0)
        abalone_data_RP_data = abalone_data_RP.fit_transform(abalone_orig_attributes)
        abalone_data_RP_df = pd.DataFrame(data = abalone_data_RP_data)

        abalone_data_RP_df_nn = pd.concat([abalone_data_RP_df,abalone_orig_labels],axis=1)

        for cluster in mycluster_aba:

            ''' kMeans clustering '''
            startTime = datetime.now()
            myk_mean_RP_prediction = KMeans(n_clusters=cluster,random_state=0).fit_predict(abalone_data_RP_df)
            myk_mean_accuracy_res = my_accuracy_cluster(abalone_orig_labels,myk_mean_RP_prediction)
            endTime = datetime.now()
            # append accuracy
            my_accuracy_kmeans_RP[RP_comp][cluster] = myk_mean_accuracy_res
            # append my_time array
            my_time_kmeans_RP[RP_comp][cluster] = (endTime-startTime).total_seconds()

            ''' EM using GaussianMixture clustering '''
            startTime = datetime.now()
            my_em_prediction = GaussianMixture(n_components=cluster).fit(abalone_data_RP_df).predict(abalone_data_RP_df)
            my_accuracy_em_res = my_accuracy_cluster(abalone_orig_labels,my_em_prediction)
            endTime = datetime.now()

            # append accuracy
            my_accuracy_em_RP[RP_comp][cluster] = my_accuracy_em_res
            # append my_time array
            my_time_em_RP[RP_comp][cluster] = (endTime-startTime).total_seconds()

        ''' NN GridSearch '''
        NN_RP_score[RP_comp] = NN_learner(abalone_data_RP_df_nn)

    plot_time_feature_transform(my_time_kmeans_RP,"k-means RP clusters vs time")
    plot_score_feature_transform(my_accuracy_kmeans_RP,"k-means RP clusters vs score")
    plot_time_feature_transform(my_time_em_RP,"EM RP clusters vs time")
    plot_score_feature_transform(my_accuracy_em_RP,"EM RP clusters vs score")

    ''' to illustrate the data in 2 component RP '''
    abalone_data_RP = GaussianRandomProjection(n_components=2)
    abalone_data_RP_data = abalone_data_RP.fit_transform(abalone_orig_attributes)
    abalone_data_RP_df = pd.DataFrame(data = abalone_data_RP_data)

    fig = plt.figure(figsize = (8,8))
    ax = fig.add_subplot(1,1,1)
    ax.set_xlabel('Component 1', fontsize = 15)
    ax.set_ylabel('Component 2', fontsize = 15)
    ax.set_title('2 component RP', fontsize = 20)
    targets = list(set(abalone_orig_labels))
    colors = ['r', 'g', 'b','c','m','y','k']
    for target, color in zip(targets,colors):
        indicesToKeep = abalone_orig_labels == target
        ax.scatter(abalone_data_RP_df.loc[indicesToKeep,0]
                   , abalone_data_RP_df.loc[indicesToKeep,1]
                   , c = color
                   , s = 50)
    ax.legend(targets)
    ax.grid()
    fig.savefig(mydir+r"\HW3_"+"data_in_2_RP"+".png")   # save the figure to file


    ''''''''''''''''''''''''''''''''''''''''''''''''''''''''''''''''''''''''''''''
    '''k - means cluster and EM (using gaussian mixture) after RFE'''
    ''''''''''''''''''''''''''''''''''''''''''''''''''''''''''''''''''''''''''''''
    # for abalone ICA, we can only have 11 Principal components since the number of features for abalone is 11
    RFE_component_abalone = [1,2,3,4,6,8]

    NN_RFE_score = defaultdict(dict)

    estimator = SVR(kernel="linear")

    my_accuracy_kmeans_RFE = defaultdict(dict)
    my_time_kmeans_RFE = defaultdict(dict)
    my_accuracy_em_RFE = defaultdict(dict)
    my_time_em_RFE = defaultdict(dict)

    for RFE_comp in RFE_component_abalone :

        abalone_data_RFE = RFE(estimator,n_features_to_select=RFE_comp)
        abalone_data_RFE_data = abalone_data_RFE.fit_transform(abalone_orig_attributes,abalone_orig_labels)
        abalone_data_RFE_df = pd.DataFrame(data = abalone_data_RFE_data)

        abalone_data_RFE_df_nn = pd.concat([abalone_data_RFE_df,abalone_orig_labels],axis=1)
        for cluster in mycluster_aba:

            ''' kMeans clustering '''
            startTime = datetime.now()
            myk_mean_RFE_prediction = KMeans(n_clusters=cluster,random_state=0).fit_predict(abalone_data_RFE_df)
            myk_mean_accuracy_res = my_accuracy_cluster(abalone_orig_labels,myk_mean_RFE_prediction)
            endTime = datetime.now()
            # append accuracy
            my_accuracy_kmeans_RFE[RFE_comp][cluster] = myk_mean_accuracy_res
            # append my_time array
            my_time_kmeans_RFE[RFE_comp][cluster] = (endTime-startTime).total_seconds()

            ''' EM using GaussianMixture clustering '''
            startTime = datetime.now()
            my_em_prediction = GaussianMixture(n_components=cluster).fit(abalone_data_RFE_df).predict(abalone_data_RFE_df)
            my_accuracy_em_res = my_accuracy_cluster(abalone_orig_labels,my_em_prediction)
            endTime = datetime.now()

            # append accuracy
            my_accuracy_em_RFE[RFE_comp][cluster] = my_accuracy_em_res
            # append my_time array
            my_time_em_RFE[RFE_comp][cluster] = (endTime-startTime).total_seconds()

        ''' NN GridSearch '''
        NN_RFE_score[RFE_comp] = NN_learner(abalone_data_RFE_df_nn)

    plot_time_feature_transform(my_time_kmeans_RFE,"k-means RFE clusters vs time")
    plot_score_feature_transform(my_accuracy_kmeans_RFE,"k-means RFE clusters vs score")
    plot_time_feature_transform(my_time_em_RFE,"EM RFE clusters vs time")
    plot_score_feature_transform(my_accuracy_em_RFE,"EM RFE clusters vs score")

    ''' to illustrate the data in 2 component RFE '''
    abalone_data_RFE = RFE(estimator,n_features_to_select=2)
    abalone_data_RFE_data = abalone_data_RFE.fit_transform(abalone_orig_attributes,abalone_orig_labels)
    abalone_data_RFE_df = pd.DataFrame(data = abalone_data_RFE_data)

    fig = plt.figure(figsize = (8,8))
    ax = fig.add_subplot(1,1,1)
    ax.set_xlabel('Component 1', fontsize = 15)
    ax.set_ylabel('Component 2', fontsize = 15)
    ax.set_title('2 component RFE', fontsize = 20)
    targets = list(set(abalone_orig_labels))
    colors = ['r', 'g', 'b','c','m','y','k']
    for target, color in zip(targets,colors):
        indicesToKeep = abalone_orig_labels == target
        ax.scatter(abalone_data_RFE_df.loc[indicesToKeep,0]
                   , abalone_data_RFE_df.loc[indicesToKeep,1]
                   , c = color
                   , s = 50)
    ax.legend(targets)
    ax.grid()
    fig.savefig(mydir+r"\HW3_"+"data_in_2_RFE"+".png")   # save the figure to file

    ''''''''''''''''''''''''''''''''''''''''''''''''''''''''''''''''''''''''''''''
    '''lastly run the best feature transformation, do a clustering with kmean and EM and rerun the NN with the data including the clustering'''
    ''''''''''''''''''''''''''''''''''''''''''''''''''''''''''''''''''''''''''''''
    ''' PCA kMeans clustering '''

    startTime = datetime.now()
    abalone_data_PCA_NN = PCA(n_components=1,random_state=0)
    abalone_data_PCA_NN_data = abalone_data_PCA_NN.fit_transform(abalone_orig_attributes)
    abalone_data_PCA_NN_df = pd.DataFrame(data = abalone_data_PCA_NN_data)

    abalone_data_PCA_NN_df_nn = pd.concat([abalone_data_PCA_NN_df,abalone_orig_labels],axis=1)

    myk_mean_PCA_NN_prediction = KMeans(n_clusters=50,random_state=0).fit_predict(abalone_data_PCA_NN_df)
    abalone_data_kmean_PCA_NN_df = pd.DataFrame(data = myk_mean_PCA_NN_prediction,columns=['clusters'])

    abalone_data_PCA_NN_df_nn = pd.concat([abalone_data_PCA_NN_df_nn,abalone_data_kmean_PCA_NN_df],axis=1)

    NN_PCA_NN_kmean_score = NN_learner(abalone_data_PCA_NN_df_nn)
    endTime = datetime.now()

    NN_PCA_NN_kmean_time = (endTime - startTime).total_seconds()

    ''' PCA EM using GaussianMixture clustering '''

    startTime = datetime.now()
    abalone_data_PCA_NN = PCA(n_components=1,random_state=0)
    abalone_data_PCA_NN_data = abalone_data_PCA_NN.fit_transform(abalone_orig_attributes)
    abalone_data_PCA_NN_df = pd.DataFrame(data = abalone_data_PCA_NN_data)

    abalone_data_PCA_NN_df_nn_em = pd.concat([abalone_data_PCA_NN_df,abalone_orig_labels],axis=1)

    my_em_prediction = GaussianMixture(n_components=50).fit(abalone_data_PCA_NN_df).predict(abalone_data_PCA_NN_df)
    abalone_data_em_PCA_NN_df = pd.DataFrame(data = my_em_prediction,columns=['clusters'])

    abalone_data_PCA_NN_df_nn_em = pd.concat([abalone_data_PCA_NN_df_nn_em,abalone_data_em_PCA_NN_df],axis=1)

    NN_PCA_NN_EM_score = NN_learner(abalone_data_PCA_NN_df_nn_em)
    endTime = datetime.now()

    NN_PCA_NN_EM_time = (endTime - startTime).total_seconds()

    ''' ICA kMeans clustering '''

    startTime = datetime.now()
    abalone_data_ICA_NN = FastICA(n_components=8,random_state=0)
    abalone_data_ICA_NN_data = abalone_data_ICA_NN.fit_transform(abalone_orig_attributes)
    abalone_data_ICA_NN_df = pd.DataFrame(data = abalone_data_ICA_NN_data)

    abalone_data_ICA_NN_df_nn = pd.concat([abalone_data_ICA_NN_df,abalone_orig_labels],axis=1)

    myk_mean_ICA_NN_prediction = KMeans(n_clusters=50,random_state=0).fit_predict(abalone_data_ICA_NN_df)
    abalone_data_kmean_ICA_NN_df = pd.DataFrame(data = myk_mean_ICA_NN_prediction,columns=['clusters'])

    abalone_data_ICA_NN_df_nn = pd.concat([abalone_data_ICA_NN_df_nn,abalone_data_kmean_ICA_NN_df],axis=1)

    NN_ICA_NN_kmean_score = NN_learner(abalone_data_ICA_NN_df_nn)
    endTime = datetime.now()

    NN_ICA_NN_kmean_time = (endTime - startTime).total_seconds()

    ''' ICA EM using GaussianMixture clustering '''

    startTime = datetime.now()
    abalone_data_ICA_NN = FastICA(n_components=8,random_state=0)
    abalone_data_ICA_NN_data = abalone_data_ICA_NN.fit_transform(abalone_orig_attributes)
    abalone_data_ICA_NN_df = pd.DataFrame(data = abalone_data_ICA_NN_data)

    abalone_data_ICA_NN_df_nn_em = pd.concat([abalone_data_ICA_NN_df,abalone_orig_labels],axis=1)

    my_em_prediction = GaussianMixture(n_components=50).fit(abalone_data_ICA_NN_df).predict(abalone_data_ICA_NN_df)
    abalone_data_em_ICA_NN_df = pd.DataFrame(data = my_em_prediction,columns=['clusters'])

    abalone_data_ICA_NN_df_nn_em = pd.concat([abalone_data_ICA_NN_df_nn_em,abalone_data_em_ICA_NN_df],axis=1)

    NN_ICA_NN_EM_score = NN_learner(abalone_data_ICA_NN_df_nn_em)
    endTime = datetime.now()

    NN_ICA_NN_EM_time = (endTime - startTime).total_seconds()

    ''' RP kMeans clustering '''

    startTime = datetime.now()
    abalone_data_RP_NN = GaussianRandomProjection(random_state=0,n_components=2)
    abalone_data_RP_NN_data = abalone_data_RP_NN.fit_transform(abalone_orig_attributes)
    abalone_data_RP_NN_df = pd.DataFrame(data = abalone_data_RP_NN_data)

    abalone_data_RP_NN_df_nn = pd.concat([abalone_data_RP_NN_df,abalone_orig_labels],axis=1)

    myk_mean_RP_NN_prediction = KMeans(n_clusters=50,random_state=0).fit_predict(abalone_data_RP_NN_df)
    abalone_data_kmean_RP_NN_df = pd.DataFrame(data = myk_mean_RP_NN_prediction,columns=['clusters'])

    abalone_data_RP_NN_df_nn = pd.concat([abalone_data_RP_NN_df_nn,abalone_data_kmean_RP_NN_df],axis=1)

    NN_RP_NN_kmean_score = NN_learner(abalone_data_RP_NN_df_nn)
    endTime = datetime.now()

    NN_RP_NN_kmean_time = (endTime - startTime).total_seconds()

    ''' RP EM using GaussianMixture clustering '''

    startTime = datetime.now()
    abalone_data_RP_NN = GaussianRandomProjection(random_state=0,n_components=2)
    abalone_data_RP_NN_data = abalone_data_RP_NN.fit_transform(abalone_orig_attributes)
    abalone_data_RP_NN_df = pd.DataFrame(data = abalone_data_RP_NN_data)

    abalone_data_RP_NN_df_nn_em = pd.concat([abalone_data_RP_NN_df,abalone_orig_labels],axis=1)

    my_em_prediction = GaussianMixture(n_components=50).fit(abalone_data_RP_NN_df).predict(abalone_data_RP_NN_df)
    abalone_data_em_RP_NN_df = pd.DataFrame(data = my_em_prediction,columns=['clusters'])

    abalone_data_RP_NN_df_nn_em = pd.concat([abalone_data_RP_NN_df_nn_em,abalone_data_em_RP_NN_df],axis=1)

    NN_RP_NN_EM_score = NN_learner(abalone_data_RP_NN_df_nn_em)
    endTime = datetime.now()

    NN_RP_NN_EM_time = (endTime - startTime).total_seconds()

    ''' RFE kMeans clustering '''

    startTime = datetime.now()
    abalone_data_RFE_NN = RFE(estimator,n_features_to_select=3)
    abalone_data_RFE_NN_data = abalone_data_RFE_NN.fit_transform(abalone_orig_attributes,abalone_orig_labels)
    abalone_data_RFE_NN_df = pd.DataFrame(data = abalone_data_RFE_NN_data)

    abalone_data_RFE_NN_df_nn = pd.concat([abalone_data_RFE_NN_df,abalone_orig_labels],axis=1)

    myk_mean_RFE_NN_prediction = KMeans(n_clusters=50,random_state=0).fit_predict(abalone_data_RFE_NN_df)
    abalone_data_kmean_RFE_NN_df = pd.DataFrame(data = myk_mean_RFE_NN_prediction,columns=['clusters'])

    abalone_data_RFE_NN_df_nn = pd.concat([abalone_data_RFE_NN_df_nn,abalone_data_kmean_RFE_NN_df],axis=1)

    NN_RFE_NN_kmean_score = NN_learner(abalone_data_RFE_NN_df_nn)
    endTime = datetime.now()

    NN_RFE_NN_kmean_time = (endTime - startTime).total_seconds()

    ''' RFE EM using GaussianMixture clustering '''

    startTime = datetime.now()
    abalone_data_RFE_NN = RFE(estimator,n_features_to_select=3)
    abalone_data_RFE_NN_data = abalone_data_RFE_NN.fit_transform(abalone_orig_attributes,abalone_orig_labels)
    abalone_data_RFE_NN_df = pd.DataFrame(data = abalone_data_RFE_NN_data)

    abalone_data_RFE_NN_df_nn_em = pd.concat([abalone_data_RFE_NN_df,abalone_orig_labels],axis=1)

    my_em_prediction = GaussianMixture(n_components=50).fit(abalone_data_RFE_NN_df).predict(abalone_data_RFE_NN_df)
    abalone_data_em_RFE_NN_df = pd.DataFrame(data = my_em_prediction,columns=['clusters'])

    abalone_data_RFE_NN_df_nn_em = pd.concat([abalone_data_RFE_NN_df_nn_em,abalone_data_em_RFE_NN_df],axis=1)

    NN_RFE_NN_EM_score = NN_learner(abalone_data_RFE_NN_df_nn_em)
    endTime = datetime.now()

    NN_RFE_NN_EM_time = (endTime - startTime).total_seconds()
