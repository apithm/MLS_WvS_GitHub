import pandas as pd
import matplotlib.pyplot as plt
import numpy as np
from sklearn.linear_model import LogisticRegression

from sklearn.model_selection import train_test_split
# from sklearn.neighbors import KNeighborsRegressor
from sklearn.neighbors import KNeighborsClassifier
from sklearn.metrics import mean_squared_error
from math import sqrt

import seaborn as sns

from sklearn.model_selection import GridSearchCV,RepeatedStratifiedKFold,StratifiedKFold

from sklearn.ensemble import GradientBoostingClassifier, RandomForestClassifier

from sklearn import svm

from sklearn.feature_selection import SelectKBest, mutual_info_classif
from sklearn.feature_selection import chi2

import time
from sklearn import preprocessing
from sklearn.naive_bayes import GaussianNB

from sklearn.feature_selection import SequentialFeatureSelector
from sklearn.pipeline import Pipeline

from sklearn.model_selection import KFold,StratifiedKFold

from sklearn.model_selection import cross_val_score
from sklearn.naive_bayes import MultinomialNB,ComplementNB,CategoricalNB,BernoulliNB
from sklearn.discriminant_analysis import LinearDiscriminantAnalysis
from sklearn.neural_network import MLPClassifier
from sklearn.discriminant_analysis import QuadraticDiscriminantAnalysis
from sklearn.ensemble import AdaBoostClassifier
from sklearn.tree import DecisionTreeClassifier

from pylab import figure, clf, plot, xlabel, ylabel, xlim, ylim, title, grid, axes, show, legend,savefig
import os.path
from scipy.stats import ttest_ind
import joblib
import time

# scaling_method = 'minmax'
scaling_method = 'std'


k_fold=10

trial=10

feature=138


level='all'

gender='mixed'

mlp=np.zeros([2,trial+1+1+1,feature+2])
adab=np.zeros([2,trial+1+1+1,feature+2])
knn=np.zeros([2,trial+1+1+1,feature+2])
knn_k=np.zeros([2,trial+1+1+1,feature+2])
lr=np.zeros([2,trial+1+1+1,feature+2])
nb=np.zeros([2,trial+1+1+1,feature+2])
rf_max=np.zeros([2,trial+1+1+1,feature+2])
rf_log2=np.zeros([2,trial+1+1+1,feature+2])
rf_sqrt=np.zeros([2,trial+1+1+1,feature+2])
svm_=np.zeros([2,trial+1+1+1,feature+2])
svm_C=np.zeros([2,trial+1+1+1,feature+2])
svm_rbf=np.zeros([2,trial+1+1+1,feature+2])
svm_rbf_C=np.zeros([2,trial+1+1+1,feature+2])
svm_rbf_gamma=np.zeros([2,trial+1+1+1,feature+2])

mlp_f1=np.zeros([2,trial+1+1+1,feature+2])
adab_f1=np.zeros([2,trial+1+1+1,feature+2])
knn_f1=np.zeros([2,trial+1+1+1,feature+2])
knn_k_f1=np.zeros([2,trial+1+1+1,feature+2])
lr_f1=np.zeros([2,trial+1+1+1,feature+2])
nb_f1=np.zeros([2,trial+1+1+1,feature+2])
rf_max_f1=np.zeros([2,trial+1+1+1,feature+2])
rf_log2_f1=np.zeros([2,trial+1+1+1,feature+2])
rf_sqrt_f1=np.zeros([2,trial+1+1+1,feature+2])
svm_f1=np.zeros([2,trial+1+1+1,feature+2])
svm_C_f1=np.zeros([2,trial+1+1+1,feature+2])
svm_rbf_f1=np.zeros([2,trial+1+1+1,feature+2])
svm_rbf_C_f1=np.zeros([2,trial+1+1+1,feature+2])
svm_rbf_gamma_f1=np.zeros([2,trial+1+1+1,feature+2])

mlp_auc=np.zeros([2,trial+1+1+1,feature+2])
adab_auc=np.zeros([2,trial+1+1+1,feature+2])
knn_auc=np.zeros([2,trial+1+1+1,feature+2])
knn_k_auc=np.zeros([2,trial+1+1+1,feature+2])
lr_auc=np.zeros([2,trial+1+1+1,feature+2])
nb_auc=np.zeros([2,trial+1+1+1,feature+2])
rf_max_auc=np.zeros([2,trial+1+1+1,feature+2])
rf_log2_auc=np.zeros([2,trial+1+1+1,feature+2])
rf_sqrt_auc=np.zeros([2,trial+1+1+1,feature+2])
svm_auc=np.zeros([2,trial+1+1+1,feature+2])
svm_C_auc=np.zeros([2,trial+1+1+1,feature+2])
svm_rbf_auc=np.zeros([2,trial+1+1+1,feature+2])
svm_rbf_C_auc=np.zeros([2,trial+1+1+1,feature+2])
svm_rbf_gamma_auc=np.zeros([2,trial+1+1+1,feature+2])

for j in range(0,2):
    for i in range(0,feature):
        mlp[j,0,i]=i+1
        adab[j,0,i]=i+1
        knn[j,0,i]=i+1
        knn_k[j,0,i]=i+1
        lr[j,0,i]=i+1
        nb[j,0,i]=i+1
        rf_max[j,0,i]=i+1
        rf_log2[j,0,i]=i+1
        rf_sqrt[j,0,i]=i+1
        svm_[j,0,i]=i+1
        svm_C[j,0,i]=i+1
        svm_rbf[j,0,i]=i+1
        svm_rbf_C[j,0,i]=i+1
        svm_rbf_gamma[j,0,i]=i+1

        mlp_f1[j,0,i]=i+1
        adab_f1[j,0,i]=i+1
        knn_f1[j,0,i]=i+1
        knn_k_f1[j,0,i]=i+1
        lr_f1[j,0,i]=i+1
        nb_f1[j,0,i]=i+1
        rf_max_f1[j,0,i]=i+1
        rf_log2_f1[j,0,i]=i+1
        rf_sqrt_f1[j,0,i]=i+1
        svm_f1[j,0,i]=i+1
        svm_C_f1[j,0,i]=i+1
        svm_rbf_f1[j,0,i]=i+1
        svm_rbf_C_f1[j,0,i]=i+1
        svm_rbf_gamma_f1[j,0,i]=i+1

        mlp_auc[j,0,i]=i+1
        adab_auc[j,0,i]=i+1
        knn_auc[j,0,i]=i+1
        knn_k_auc[j,0,i]=i+1
        lr_auc[j,0,i]=i+1
        nb_auc[j,0,i]=i+1
        rf_max_auc[j,0,i]=i+1
        rf_log2_auc[j,0,i]=i+1
        rf_sqrt_auc[j,0,i]=i+1
        svm_auc[j,0,i]=i+1
        svm_C_auc[j,0,i]=i+1
        svm_rbf_auc[j,0,i]=i+1
        svm_rbf_C_auc[j,0,i]=i+1
        svm_rbf_gamma_auc[j,0,i]=i+1



for condition_count in range(0,1):

    if condition_count==0:
        condition = 'control'
    else:
        condition='baseline'


    prefix = 'mixed features_control_exp_class_' + level + '_'   

    dataset_train = 'ECG_EEG_' + gender + '_' + condition + '_exp_' + level + '_train'
    # print(dataset)


    path = 'D:\\Dropbox\\NECTEC\\My Documents\\StressClassification_NRIIS\\ECG & EEG_2_1\\'
    filename = dataset_train + '.csv'
    fullname = path + filename

    savepath='D:\\Dropbox\\NECTEC\\My Documents\\StressClassification_NRIIS\\Result\\Full_ECG_EEG_2types_allclass_' + gender +'_gender_10trialsMSI_50fea\\'
    if os.path.isdir(savepath)==False:
            os.makedirs(savepath)

    savepath_model=savepath+'\\model\\' + condition + '\\' + level + '\\'
    if os.path.isdir(savepath_model)==False:
            os.makedirs(savepath_model)


    # if condition=='EO_AC1' or condition=='EO_AC2':
    #     prefix = 'arith_stdzminmax_2class_filter_'

    # elif condition=='EO_AC1_AC2':
    #     prefix = 'arith_stdzminmax_3class_filter_'
        
    # elif condition=='EO_AC1AC2':



    data = pd.read_csv(fullname,header=0)


    X = data.drop('Class',axis=1).values
    y = data['Class'].values


    temp=data.shape
    data_len=temp[0]
    feature_all=temp[1]-1



    # print(adab)
    # n_features_selected=list()
    # n_features_selected.append('No. of features')

    selected_feature=np.zeros([feature,feature])
    ranked_feature=list()
    # -------------------
    n_features_selected=list()
    n_features_selected.append('No. of features')


    # MinMaxScaler = preprocessing.MinMaxScaler()
    # X = MinMaxScaler.fit_transform(X)

    StdScaler = preprocessing.StandardScaler()
    X = StdScaler.fit_transform(X)


    N_features=feature
    # N_features=

    # tic = time.perf_counter()
    for trial_count in range(1,trial+1):
    # for feature_count in range(1,feature+1):
        
        for feature_count in range(1,N_features+1):
            
            print(dataset_train + ' ' + str(trial_count) + ' ' + str(feature_count)) 
            
            # print(dataset)
            # print(feature_count)

            # if condition_count==0:
            n_features_selected.append(feature_count)
            # Feature extraction
            # tic = time.perf_counter()
            test = SelectKBest(k=feature_count)
            fit = test.fit(X, y)

            selected_X = fit.transform(X)

            # toc = time.perf_counter()

            # filter_time=toc-tic
            # Summarize selected features
            np.set_printoptions(precision=3)
            # print(fit.scores_)
            # print(selected_X)

            dim=selected_X.shape
        
            k=0
            for i in range(dim[1]):
                for j in range(feature_all):
                    if np.array_equal(selected_X[:,i],X[:,j]):
                        selected_feature[feature_count-1,k]=j+1
                        if (j+1 not in ranked_feature):
                            ranked_feature.append(j+1)
                        k=k+1
                        ranked_feature_df = pd.DataFrame ([ranked_feature])
            # ------------------


            # 
            
            # # ----------MLP-------------

            cv = StratifiedKFold(n_splits=k_fold, shuffle=True)
            # create model
            # model = AdaBoostClassifier(n_estimators=100)
            model = MLPClassifier(max_iter=10000,hidden_layer_sizes=[np.int32(np.floor((feature_count+2)/2)),np.int32(np.floor((feature_count+2)/2))])

            scores = cross_val_score(model, selected_X, y, scoring='accuracy', cv=cv, n_jobs=-1)

            mlp[condition_count,trial_count,feature_count-1]=np.mean(scores)*100

            scores = cross_val_score(model, selected_X, y, scoring='f1_macro', cv=cv, n_jobs=-1)

            mlp_f1[condition_count,trial_count,feature_count-1]=np.mean(scores)

            scores = cross_val_score(model, selected_X, y, scoring='roc_auc_ovr', cv=cv, n_jobs=-1)

            mlp_auc[condition_count,trial_count,feature_count-1]=np.mean(scores)

            # if trial_count==1:
            #     model_filename = savepath_model + 'mlp_' + str(feature_count) +  ".joblib"
            #     joblib.dump(model, model_filename)

            # # ----------ADABOOST-------------
            # prepare the cross-validation procedure

            cv = StratifiedKFold(n_splits=k_fold, shuffle=True)
            # create model
            model = AdaBoostClassifier(n_estimators=100)

            scores = cross_val_score(model, selected_X, y, scoring='accuracy', cv=cv, n_jobs=-1)

            adab[condition_count,trial_count,feature_count-1]=np.mean(scores)*100

            scores = cross_val_score(model, selected_X, y, scoring='f1_macro', cv=cv, n_jobs=-1)

            adab_f1[condition_count,trial_count,feature_count-1]=np.mean(scores)

            scores = cross_val_score(model, selected_X, y, scoring='roc_auc_ovr', cv=cv, n_jobs=-1)

            adab_auc[condition_count,trial_count,feature_count-1]=np.mean(scores)
        
            # if trial_count==1:
            #     model_filename = savepath_model + 'adab_' + str(feature_count) +  ".joblib"
            #     joblib.dump(model, model_filename)

            # ----------KNN-------------
            cv = StratifiedKFold(n_splits=k_fold, shuffle=True)
            parameters = {"n_neighbors": range(1, np.int32(data_len*((k_fold-1)/10)),2)}
            # print(np.int32(data_len*((k_fold-1)/10)))

            gridsearch = GridSearchCV(KNeighborsClassifier(), parameters, scoring='accuracy', cv=cv,n_jobs=-1)
            gridsearch.fit(selected_X, y)
    
        
            knn[condition_count,trial_count,feature_count-1]=gridsearch.best_score_*100
            
            temp=gridsearch.best_params_
            knn_k[condition_count,trial_count,feature_count-1]=temp['n_neighbors']


            gridsearch = GridSearchCV(KNeighborsClassifier(), parameters, scoring='f1_macro', cv=cv,n_jobs=-1)
            gridsearch.fit(selected_X, y)
    
        
            knn_f1[condition_count,trial_count,feature_count-1]=gridsearch.best_score_
            
            temp=gridsearch.best_params_
            knn_k_f1[condition_count,trial_count,feature_count-1]=temp['n_neighbors']

            
            gridsearch = GridSearchCV(KNeighborsClassifier(), parameters, scoring='roc_auc_ovr', cv=cv,n_jobs=-1)
            gridsearch.fit(selected_X, y)
    
        
            knn_auc[condition_count,trial_count,feature_count-1]=gridsearch.best_score_
            
            temp=gridsearch.best_params_
            knn_k_auc[condition_count,trial_count,feature_count-1]=temp['n_neighbors']

            # if trial_count==1:
            #     model_filename = savepath_model + 'knn_' + str(feature_count) +  ".joblib"
                # joblib.dump(model, model_filename)

            # ----------Logistic Regression-------------
    
            cv = StratifiedKFold(n_splits=k_fold, shuffle=True)

            model = LogisticRegression()
            # evaluate model
            scores = cross_val_score(model, selected_X, y, scoring='accuracy', cv=cv, n_jobs=-1)

            lr[condition_count,trial_count,feature_count-1]=np.mean(scores)*100

            scores = cross_val_score(model, selected_X, y, scoring='f1_macro', cv=cv, n_jobs=-1)

            lr_f1[condition_count,trial_count,feature_count-1]=np.mean(scores)

            scores = cross_val_score(model, selected_X, y, scoring='roc_auc_ovr', cv=cv, n_jobs=-1)

            lr_auc[condition_count,trial_count,feature_count-1]=np.mean(scores)

            # if trial_count==1:
            #     model_filename = savepath_model + 'lr_' + str(feature_count) +  ".joblib"
            #     joblib.dump(model, model_filename)


            # ----------Naive Bayes-------------
            # prepare the cross-validation procedure

            cv = StratifiedKFold(n_splits=k_fold, shuffle=True)
            # create model
            model = GaussianNB()

            scores = cross_val_score(model, selected_X, y, scoring='accuracy', cv=cv, n_jobs=-1)

            nb[condition_count,trial_count,feature_count-1]=np.mean(scores)*100

            scores = cross_val_score(model, selected_X, y, scoring='f1_macro', cv=cv, n_jobs=-1)

            nb_f1[condition_count,trial_count,feature_count-1]=np.mean(scores)

            scores = cross_val_score(model, selected_X, y, scoring='roc_auc_ovr', cv=cv, n_jobs=-1)

            nb_auc[condition_count,trial_count,feature_count-1]=np.mean(scores)
            # print(scores)

            # if trial_count==1:
            #     model_filename = savepath_model + 'nb_' + str(feature_count) +  ".joblib"
            #     joblib.dump(model, model_filename)
        
            # ----------Max RF------------
            cv = StratifiedKFold(n_splits=k_fold, shuffle=True)
            parameters = {"max_features": [None]} #If None, then max_features=n_features
            gridsearch = GridSearchCV(RandomForestClassifier(n_estimators=100), parameters,cv=cv, scoring='accuracy',n_jobs=-1)
            gridsearch.fit(selected_X,y)
        
            rf_max[condition_count,trial_count,feature_count-1]=gridsearch.best_score_*100

            gridsearch = GridSearchCV(RandomForestClassifier(n_estimators=100), parameters,cv=cv, scoring='f1_macro',n_jobs=-1)
            gridsearch.fit(selected_X,y)
        
            rf_max_f1[condition_count,trial_count,feature_count-1]=gridsearch.best_score_

            gridsearch = GridSearchCV(RandomForestClassifier(n_estimators=100), parameters,cv=cv, scoring='roc_auc_ovr',n_jobs=-1)
            gridsearch.fit(selected_X,y)
        
            rf_max_auc[condition_count,trial_count,feature_count-1]=gridsearch.best_score_
            
            # if trial_count==1:
            #     model_filename = savepath_model + 'rf_max_' + str(feature_count) +  ".joblib"
            #     joblib.dump(model, model_filename)

            # ----------log2 RF------------
            cv = StratifiedKFold(n_splits=k_fold, shuffle=True)
            parameters = {"max_features": ['log2']} #If None, then max_features=n_features
            gridsearch = GridSearchCV(RandomForestClassifier(n_estimators=100), parameters,cv=cv,n_jobs=-1)
            gridsearch.fit(selected_X,y)

            rf_log2[condition_count,trial_count,feature_count-1]=gridsearch.best_score_*100

            gridsearch = GridSearchCV(RandomForestClassifier(n_estimators=100), parameters,cv=cv, scoring='f1_macro',n_jobs=-1)
            gridsearch.fit(selected_X,y)

            rf_log2_f1[condition_count,trial_count,feature_count-1]=gridsearch.best_score_

            gridsearch = GridSearchCV(RandomForestClassifier(n_estimators=100), parameters,cv=cv, scoring='roc_auc_ovr',n_jobs=-1)
            gridsearch.fit(selected_X,y)
        
            rf_log2_auc[condition_count,trial_count,feature_count-1]=gridsearch.best_score_

            # if trial_count==1:
            #     model_filename = savepath_model + 'rf_log2_' + str(feature_count) +  ".joblib"
            #     joblib.dump(model, model_filename)

            # ----------Sqrt RF------------
            cv = StratifiedKFold(n_splits=k_fold, shuffle=True)
            parameters = {"max_features": ['sqrt']} #If None, then max_features=n_features
            gridsearch = GridSearchCV(RandomForestClassifier(n_estimators=100), parameters,cv=cv,n_jobs=-1)
            gridsearch.fit(selected_X,y)

            rf_sqrt[condition_count,trial_count,feature_count-1]=gridsearch.best_score_*100

            gridsearch = GridSearchCV(RandomForestClassifier(n_estimators=100), parameters,cv=cv, scoring='f1_macro',n_jobs=-1)
            gridsearch.fit(selected_X,y)
        
            rf_sqrt_f1[condition_count,trial_count,feature_count-1]=gridsearch.best_score_

            gridsearch = GridSearchCV(RandomForestClassifier(n_estimators=100), parameters,cv=cv, scoring='roc_auc_ovr',n_jobs=-1)
            gridsearch.fit(selected_X,y)
        
            rf_sqrt_auc[condition_count,trial_count,feature_count-1]=gridsearch.best_score_

            # if trial_count==1:
            #     model_filename = savepath_model + 'rf_sqrt_' + str(feature_count) +  ".joblib"
            #     joblib.dump(model, model_filename)

            # ----------SVM------------
            cv = StratifiedKFold(n_splits=k_fold, shuffle=True)
            parameters = {"C": [0.0001,0.001,0.01,0.1,1,10,100,1000,10000]}
            gridsearch = GridSearchCV(svm.SVC(kernel='linear',probability=True), parameters,cv=cv, scoring='accuracy',n_jobs=-1)
            gridsearch.fit(selected_X,y)

            svm_[condition_count,trial_count,feature_count-1]=gridsearch.best_score_*100

            temp=gridsearch.best_params_
            svm_C[condition_count,trial_count,feature_count-1]=temp['C']


            gridsearch = GridSearchCV(svm.SVC(kernel='linear',probability=True), parameters,cv=cv, scoring='f1_macro',n_jobs=-1)
            gridsearch.fit(selected_X,y)

            svm_f1[condition_count,trial_count,feature_count-1]=gridsearch.best_score_

            temp=gridsearch.best_params_
            svm_f1[condition_count,trial_count,feature_count-1]=temp['C']
    

            gridsearch = GridSearchCV(svm.SVC(kernel='linear',probability=True), parameters,cv=cv, scoring='roc_auc_ovr',n_jobs=-1)
            gridsearch.fit(selected_X,y)

            svm_auc[condition_count,trial_count,feature_count-1]=gridsearch.best_score_

            temp=gridsearch.best_params_
            svm_auc[condition_count,trial_count,feature_count-1]=temp['C']
        
            # if trial_count==1:
            #     model_filename = savepath_model + 'svm_' + str(feature_count) +  ".joblib"
            #     joblib.dump(model, model_filename)

            # ---------RBF SVM-------------
            cv = StratifiedKFold(n_splits=k_fold, shuffle=True)
            parameters = {"C": [0.0001,0.001,0.01,0.1,1,10,100,1000,10000],"gamma": [0.0001,0.001,0.01,0.1, 1]}
            gridsearch = GridSearchCV(svm.SVC(kernel='rbf',probability=True), parameters,cv=cv, scoring='accuracy',n_jobs=-1)
            gridsearch.fit(selected_X,y)

            svm_rbf[condition_count,trial_count,feature_count-1]=gridsearch.best_score_*100

            temp=gridsearch.best_params_
            svm_rbf_C[condition_count,trial_count,feature_count-1]=temp['C']
            svm_rbf_gamma[condition_count,trial_count,feature_count-1]=temp['gamma']


            gridsearch = GridSearchCV(svm.SVC(kernel='rbf',probability=True), parameters,cv=cv, scoring='f1_macro',n_jobs=-1)
            gridsearch.fit(selected_X,y)

            svm_rbf_f1[condition_count,trial_count,feature_count-1]=gridsearch.best_score_

            temp=gridsearch.best_params_
            svm_rbf_C_f1[condition_count,trial_count,feature_count-1]=temp['C']
            svm_rbf_gamma_f1[condition_count,trial_count,feature_count-1]=temp['gamma']

            
            gridsearch = GridSearchCV(svm.SVC(kernel='rbf',probability=True), parameters,cv=cv, scoring='roc_auc_ovr',n_jobs=-1)
            gridsearch.fit(selected_X,y)

            svm_rbf_auc[condition_count,trial_count,feature_count-1]=gridsearch.best_score_

            temp=gridsearch.best_params_
            svm_rbf_C_auc[condition_count,trial_count,feature_count-1]=temp['C']
            svm_rbf_gamma_auc[condition_count,trial_count,feature_count-1]=temp['gamma']

            # if trial_count==1:
            #     model_filename = savepath_model + 'svm_rbf_' + str(feature_count) +  ".joblib"
            #     joblib.dump(model, model_filename)

            ranked_feature_df_tranposed=ranked_feature_df.transpose()

            # toc = time.perf_counter()
            # print(toc)
            # print(toc-tic)

    for feature_count in range(1,N_features+1):
    
        mlp[condition_count,trial+1,feature_count-1]=np.mean(mlp[condition_count,1:trial+1,feature_count-1])
        adab[condition_count,trial+1,feature_count-1]=np.mean(adab[condition_count,1:trial+1,feature_count-1])
        knn[condition_count,trial+1,feature_count-1]=np.mean(knn[condition_count,1:trial+1,feature_count-1])
        lr[condition_count,trial+1,feature_count-1]=np.mean(lr[condition_count,1:trial+1,feature_count-1])
        nb[condition_count,trial+1,feature_count-1]=np.mean(nb[condition_count,1:trial+1,feature_count-1])
        rf_max[condition_count,trial+1,feature_count-1]=np.mean(rf_max[condition_count,1:trial+1,feature_count-1])
        rf_log2[condition_count,trial+1,feature_count-1]=np.mean(rf_log2[condition_count,1:trial+1,feature_count-1])
        rf_sqrt[condition_count,trial+1,feature_count-1]=np.mean(rf_sqrt[condition_count,1:trial+1,feature_count-1])
        svm_[condition_count,trial+1,feature_count-1]=np.mean(svm_[condition_count,1:trial+1,feature_count-1])
        svm_rbf[condition_count,trial+1,feature_count-1]=np.mean(svm_rbf[condition_count,1:trial+1,feature_count-1])


        mlp[condition_count,trial+2,feature_count-1]=np.std(mlp[condition_count,1:trial+1,feature_count-1])
        adab[condition_count,trial+2,feature_count-1]=np.std(adab[condition_count,1:trial+1,feature_count-1])
        knn[condition_count,trial+2,feature_count-1]=np.std(knn[condition_count,1:trial+1,feature_count-1])
        lr[condition_count,trial+2,feature_count-1]=np.std(lr[condition_count,1:trial+1,feature_count-1])
        nb[condition_count,trial+2,feature_count-1]=np.std(nb[condition_count,1:trial+1,feature_count-1])
        rf_max[condition_count,trial+2,feature_count-1]=np.std(rf_max[condition_count,1:trial+1,feature_count-1])
        rf_log2[condition_count,trial+2,feature_count-1]=np.std(rf_log2[condition_count,1:trial+1,feature_count-1])
        rf_sqrt[condition_count,trial+2,feature_count-1]=np.std(rf_sqrt[condition_count,1:trial+1,feature_count-1])
        svm_[condition_count,trial+2,feature_count-1]=np.std(svm_[condition_count,1:trial+1,feature_count-1])
        svm_rbf[condition_count,trial+2,feature_count-1]=np.std(svm_rbf[condition_count,1:trial+1,feature_count-1])


        mlp_f1[condition_count,trial+1,feature_count-1]=np.mean(mlp_f1[condition_count,1:trial+1,feature_count-1])
        adab_f1[condition_count,trial+1,feature_count-1]=np.mean(adab_f1[condition_count,1:trial+1,feature_count-1])
        knn_f1[condition_count,trial+1,feature_count-1]=np.mean(knn_f1[condition_count,1:trial+1,feature_count-1])
        lr_f1[condition_count,trial+1,feature_count-1]=np.mean(lr_f1[condition_count,1:trial+1,feature_count-1])
        nb_f1[condition_count,trial+1,feature_count-1]=np.mean(nb_f1[condition_count,1:trial+1,feature_count-1])
        rf_max_f1[condition_count,trial+1,feature_count-1]=np.mean(rf_max_f1[condition_count,1:trial+1,feature_count-1])
        rf_log2_f1[condition_count,trial+1,feature_count-1]=np.mean(rf_log2_f1[condition_count,1:trial+1,feature_count-1])
        rf_sqrt_f1[condition_count,trial+1,feature_count-1]=np.mean(rf_sqrt_f1[condition_count,1:trial+1,feature_count-1])
        svm_f1[condition_count,trial+1,feature_count-1]=np.mean(svm_f1[condition_count,1:trial+1,feature_count-1])
        svm_rbf_f1[condition_count,trial+1,feature_count-1]=np.mean(svm_rbf_f1[condition_count,1:trial+1,feature_count-1])


        mlp_f1[condition_count,trial+2,feature_count-1]=np.std(mlp_f1[condition_count,1:trial+1,feature_count-1])
        adab_f1[condition_count,trial+2,feature_count-1]=np.std(adab_f1[condition_count,1:trial+1,feature_count-1])
        knn_f1[condition_count,trial+2,feature_count-1]=np.std(knn_f1[condition_count,1:trial+1,feature_count-1])
        lr_f1[condition_count,trial+2,feature_count-1]=np.std(lr_f1[condition_count,1:trial+1,feature_count-1])
        nb_f1[condition_count,trial+2,feature_count-1]=np.std(nb_f1[condition_count,1:trial+1,feature_count-1])
        rf_max_f1[condition_count,trial+2,feature_count-1]=np.std(rf_max_f1[condition_count,1:trial+1,feature_count-1])
        rf_log2_f1[condition_count,trial+2,feature_count-1]=np.std(rf_log2_f1[condition_count,1:trial+1,feature_count-1])
        rf_sqrt_f1[condition_count,trial+2,feature_count-1]=np.std(rf_sqrt_f1[condition_count,1:trial+1,feature_count-1])
        svm_f1[condition_count,trial+2,feature_count-1]=np.std(svm_f1[condition_count,1:trial+1,feature_count-1])
        svm_rbf_f1[condition_count,trial+2,feature_count-1]=np.std(svm_rbf_f1[condition_count,1:trial+1,feature_count-1])   


        mlp_auc[condition_count,trial+1,feature_count-1]=np.mean(mlp_auc[condition_count,1:trial+1,feature_count-1])
        adab_auc[condition_count,trial+1,feature_count-1]=np.mean(adab_auc[condition_count,1:trial+1,feature_count-1])
        knn_auc[condition_count,trial+1,feature_count-1]=np.mean(knn_auc[condition_count,1:trial+1,feature_count-1])
        lr_auc[condition_count,trial+1,feature_count-1]=np.mean(lr_auc[condition_count,1:trial+1,feature_count-1])
        nb_auc[condition_count,trial+1,feature_count-1]=np.mean(nb_auc[condition_count,1:trial+1,feature_count-1])
        rf_max_auc[condition_count,trial+1,feature_count-1]=np.mean(rf_max_auc[condition_count,1:trial+1,feature_count-1])
        rf_log2_auc[condition_count,trial+1,feature_count-1]=np.mean(rf_log2_auc[condition_count,1:trial+1,feature_count-1])
        rf_sqrt_auc[condition_count,trial+1,feature_count-1]=np.mean(rf_sqrt_auc[condition_count,1:trial+1,feature_count-1])
        svm_auc[condition_count,trial+1,feature_count-1]=np.mean(svm_auc[condition_count,1:trial+1,feature_count-1])
        svm_rbf_auc[condition_count,trial+1,feature_count-1]=np.mean(svm_rbf_auc[condition_count,1:trial+1,feature_count-1])


        mlp_auc[condition_count,trial+2,feature_count-1]=np.std(mlp_auc[condition_count,1:trial+1,feature_count-1])
        adab_auc[condition_count,trial+2,feature_count-1]=np.std(adab_auc[condition_count,1:trial+1,feature_count-1])
        knn_auc[condition_count,trial+2,feature_count-1]=np.std(knn_auc[condition_count,1:trial+1,feature_count-1])
        lr_auc[condition_count,trial+2,feature_count-1]=np.std(lr_auc[condition_count,1:trial+1,feature_count-1])
        nb_auc[condition_count,trial+2,feature_count-1]=np.std(nb_auc[condition_count,1:trial+1,feature_count-1])
        rf_max_auc[condition_count,trial+2,feature_count-1]=np.std(rf_max_auc[condition_count,1:trial+1,feature_count-1])
        rf_log2_auc[condition_count,trial+2,feature_count-1]=np.std(rf_log2_auc[condition_count,1:trial+1,feature_count-1])
        rf_sqrt_auc[condition_count,trial+2,feature_count-1]=np.std(rf_sqrt_auc[condition_count,1:trial+1,feature_count-1])
        svm_auc[condition_count,trial+2,feature_count-1]=np.std(svm_auc[condition_count,1:trial+1,feature_count-1])
        svm_rbf_auc[condition_count,trial+2,feature_count-1]=np.std(svm_rbf_auc[condition_count,1:trial+1,feature_count-1])

    # if condition_count==0:

    mlp[condition_count,trial+1,N_features]=max(mlp[condition_count,trial+1,0:N_features])
    temp_mlp=np.where(mlp[condition_count,trial+1,0:N_features]==max(mlp[condition_count,trial+1,0:N_features]))
    mlp[condition_count,trial+1,N_features+1]=temp_mlp[0][0]+1

    adab[condition_count,trial+1,N_features]=max(adab[condition_count,trial+1,0:N_features])
    temp_adab=np.where(adab[condition_count,trial+1,0:N_features]==max(adab[condition_count,trial+1,0:N_features]))
    adab[condition_count,trial+1,N_features+1]=temp_adab[0][0]+1

    knn[condition_count,trial+1,N_features]=max(knn[condition_count,trial+1,0:N_features])
    temp_knn=np.where(knn[condition_count,trial+1,0:N_features]==max(knn[condition_count,trial+1,0:N_features]))
    knn[condition_count,trial+1,N_features+1]=temp_knn[0][0]+1

    lr[condition_count,trial+1,N_features]=max(lr[condition_count,trial+1,0:N_features])
    temp_lr=np.where(lr[condition_count,trial+1,0:N_features]==max(lr[condition_count,trial+1,0:N_features]))
    lr[condition_count,trial+1,N_features+1]=temp_lr[0][0]+1

    nb[condition_count,trial+1,N_features]=max(nb[condition_count,trial+1,0:N_features])
    temp_nb=np.where(nb[condition_count,trial+1,0:N_features]==max(nb[condition_count,trial+1,0:N_features]))
    nb[condition_count,trial+1,N_features+1]=temp_nb[0][0]+1

    rf_max[condition_count,trial+1,N_features]=max(rf_max[condition_count,trial+1,0:N_features])
    temp_rf_max=np.where(rf_max[condition_count,trial+1,0:N_features]==max(rf_max[condition_count,trial+1,0:N_features]))
    rf_max[condition_count,trial+1,N_features+1]=temp_rf_max[0][0]+1

    rf_log2[condition_count,trial+1,N_features]=max(rf_log2[condition_count,trial+1,0:N_features])
    temp_rf_log2=np.where(rf_log2[condition_count,trial+1,0:N_features]==max(rf_log2[condition_count,trial+1,0:N_features]))
    rf_log2[condition_count,trial+1,N_features+1]=temp_rf_log2[0][0]+1

    rf_sqrt[condition_count,trial+1,N_features]=max(rf_sqrt[condition_count,trial+1,0:N_features])
    temp_rf_sqrt=np.where(rf_sqrt[condition_count,trial+1,0:N_features]==max(rf_sqrt[condition_count,trial+1,0:N_features]))
    rf_sqrt[condition_count,trial+1,N_features+1]=temp_rf_sqrt[0][0]+1

    svm_[condition_count,trial+1,N_features]=max(svm_[condition_count,trial+1,0:N_features])
    temp_svm_=np.where(svm_[condition_count,trial+1,0:N_features]==max(svm_[condition_count,trial+1,0:N_features]))
    svm_[condition_count,trial+1,N_features+1]=temp_svm_[0][0]+1

    svm_rbf[condition_count,trial+1,N_features]=max(svm_rbf[condition_count,trial+1,0:N_features])
    temp_svm_rbf=np.where(svm_rbf[condition_count,trial+1,0:N_features]==max(svm_rbf[condition_count,trial+1,0:N_features]))
    svm_rbf[condition_count,trial+1,N_features+1]=temp_svm_rbf[0][0]+1



    # else:

    #     mlp[condition_count,trial+1,N_features]=max(mlp[condition_count,trial+1,0:N_features])
    #     temp_mlp_2=np.where(mlp[condition_count,trial+1,0:N_features]==max(mlp[condition_count,trial+1,0:N_features]))
    #     mlp[condition_count,trial+1,N_features+1]=temp_mlp_2[0][0]+1

    #     adab[condition_count,trial+1,N_features]=max(adab[condition_count,trial+1,0:N_features])
    #     temp_adab_2=np.where(adab[condition_count,trial+1,0:N_features]==max(adab[condition_count,trial+1,0:N_features]))
    #     adab[condition_count,trial+1,N_features+1]=temp_adab_2[0][0]+1

    #     knn[condition_count,trial+1,N_features]=max(knn[condition_count,trial+1,0:N_features])
    #     temp_knn_2=np.where(knn[condition_count,trial+1,0:N_features]==max(knn[condition_count,trial+1,0:N_features]))
    #     knn[condition_count,trial+1,N_features+1]=temp_knn_2[0][0]+1

    #     lr[condition_count,trial+1,N_features]=max(lr[condition_count,trial+1,0:N_features])
    #     temp_lr_2=np.where(lr[condition_count,trial+1,0:N_features]==max(lr[condition_count,trial+1,0:N_features]))
    #     lr[condition_count,trial+1,N_features+1]=temp_lr_2[0][0]+1

    #     nb[condition_count,trial+1,N_features]=max(nb[condition_count,trial+1,0:N_features])
    #     temp_nb_2=np.where(nb[condition_count,trial+1,0:N_features]==max(nb[condition_count,trial+1,0:N_features]))
    #     nb[condition_count,trial+1,N_features+1]=temp_nb_2[0][0]+1

    #     rf_max[condition_count,trial+1,N_features]=max(rf_max[condition_count,trial+1,0:N_features])
    #     temp_rf_max_2=np.where(rf_max[condition_count,trial+1,0:N_features]==max(rf_max[condition_count,trial+1,0:N_features]))
    #     rf_max[condition_count,trial+1,N_features+1]=temp_rf_max_2[0][0]+1

    #     rf_log2[condition_count,trial+1,N_features]=max(rf_log2[condition_count,trial+1,0:N_features])
    #     temp_rf_log2_2=np.where(rf_log2[condition_count,trial+1,0:N_features]==max(rf_log2[condition_count,trial+1,0:N_features]))
    #     rf_log2[condition_count,trial+1,N_features+1]=temp_rf_log2_2[0][0]+1

    #     rf_sqrt[condition_count,trial+1,N_features]=max(rf_sqrt[condition_count,trial+1,0:N_features])
    #     temp_rf_sqrt_2=np.where(rf_sqrt[condition_count,trial+1,0:N_features]==max(rf_sqrt[condition_count,trial+1,0:N_features]))
    #     rf_sqrt[condition_count,trial+1,N_features+1]=temp_rf_sqrt_2[0][0]+1

    #     svm_[condition_count,trial+1,N_features]=max(svm_[condition_count,trial+1,0:N_features])
    #     temp_svm_2=np.where(svm_[condition_count,trial+1,0:N_features]==max(svm_[condition_count,trial+1,0:N_features]))
    #     svm_[condition_count,trial+1,N_features+1]=temp_svm_2[0][0]+1

    #     svm_rbf[condition_count,trial+1,N_features]=max(svm_rbf[condition_count,trial+1,0:N_features])
    #     temp_svm_rbf_2=np.where(svm_rbf[condition_count,trial+1,0:N_features]==max(svm_rbf[condition_count,trial+1,0:N_features]))
    #     svm_rbf[condition_count,trial+1,N_features+1]=temp_svm_rbf_2[0][0]+1

    mlp_df = pd.DataFrame(mlp[condition_count,:,:])
    adab_df = pd.DataFrame(adab[condition_count,:,:])
    knn_df = pd.DataFrame(knn[condition_count,:,:])
    knn_k_df = pd.DataFrame(knn_k[condition_count,:,:])
    lr_df = pd.DataFrame(lr[condition_count,:,:])
    nb_df = pd.DataFrame(nb[condition_count,:,:])
    rf_max_df = pd.DataFrame(rf_max[condition_count,:,:])
    rf_sqrt_df = pd.DataFrame(rf_sqrt[condition_count,:,:])
    rf_log2_df = pd.DataFrame(rf_log2[condition_count,:,:])
    rf_max_df = pd.DataFrame(rf_max[condition_count,:,:])
    svm_df = pd.DataFrame(svm_[condition_count,:,:])
    svm_C_df = pd.DataFrame(svm_C[condition_count,:,:])
    svm_rbf_df = pd.DataFrame(svm_rbf[condition_count,:,:])
    svm_rbf_C_df = pd.DataFrame(svm_rbf_C[condition_count,:,:])
    svm_rbf_gamma_df = pd.DataFrame(svm_rbf_gamma[condition_count,:,:])



    ranked_feature_df_tranposed.to_excel(savepath + prefix + 'Ranked_features.xlsx',index=False,header=False)
    mlp_df.to_excel(savepath + prefix + 'mlp.xlsx',index=False,header=False)
    adab_df.to_excel(savepath + prefix + 'adab.xlsx',index=False,header=False)
    knn_df.to_excel(savepath + prefix + 'kNN.xlsx',index=False,header=False)
    knn_k_df.to_excel(savepath + prefix + 'kNN_k.xlsx',index=False,header=False)
    lr_df.to_excel(savepath + prefix + 'lr.xlsx',index=False,header=False)
    nb_df.to_excel(savepath + prefix + 'nb.xlsx',index=False,header=False)
    rf_max_df.to_excel(savepath + prefix + 'rf_max.xlsx',index=False,header=False)
    rf_sqrt_df.to_excel(savepath + prefix + 'rf_sqrt.xlsx',index=False,header=False)
    rf_log2_df.to_excel(savepath + prefix + 'rf_log2.xlsx',index=False,header=False)
    svm_df.to_excel(savepath + prefix + 'svm.xlsx',index=False,header=False)
    svm_C_df.to_excel(savepath + prefix + 'svm_C.xlsx',index=False,header=False)
    svm_rbf_df.to_excel(savepath + prefix + 'svm_rbf.xlsx',index=False,header=False)
    svm_rbf_C_df.to_excel(savepath + prefix + 'svm_rbf_C.xlsx',index=False,header=False)
    svm_rbf_gamma_df.to_excel(savepath + prefix + 'svm_rbf_gamma.xlsx',index=False,header=False)


    mlp_f1_df = pd.DataFrame(mlp_f1[condition_count,:,:])
    adab_f1_df = pd.DataFrame(adab_f1[condition_count,:,:])
    knn_f1_df = pd.DataFrame(knn_f1[condition_count,:,:])
    knn_k_f1_df = pd.DataFrame(knn_k_f1[condition_count,:,:])
    lr_f1_df = pd.DataFrame(lr_f1[condition_count,:,:])
    nb_f1_df = pd.DataFrame(nb_f1[condition_count,:,:])
    rf_max_f1_df = pd.DataFrame(rf_max_f1[condition_count,:,:])
    rf_sqrt_f1_df = pd.DataFrame(rf_sqrt_f1[condition_count,:,:])
    rf_log2_f1_df = pd.DataFrame(rf_log2_f1[condition_count,:,:])
    rf_max_f1_df = pd.DataFrame(rf_max_f1[condition_count,:,:])
    svm_f1_df = pd.DataFrame(svm_f1[condition_count,:,:])
    svm_C_f1_df = pd.DataFrame(svm_C_f1[condition_count,:,:])
    svm_rbf_f1_df = pd.DataFrame(svm_rbf_f1[condition_count,:,:])
    svm_rbf_C_f1_df = pd.DataFrame(svm_rbf_C_f1[condition_count,:,:])
    svm_rbf_gamma_f1_df = pd.DataFrame(svm_rbf_gamma_f1[condition_count,:,:])



    # ranked_feature_f1_df_tranposed.to_excel(savepath + prefix + 'Ranked_features.xlsx',index=False,header=False)
    mlp_f1_df.to_excel(savepath + prefix + 'mlp_f1.xlsx',index=False,header=False)
    adab_f1_df.to_excel(savepath + prefix + 'adab_f1.xlsx',index=False,header=False)
    knn_f1_df.to_excel(savepath + prefix + 'kNN_f1.xlsx',index=False,header=False)
    knn_k_f1_df.to_excel(savepath + prefix + 'kNN_k_f1.xlsx',index=False,header=False)
    lr_f1_df.to_excel(savepath + prefix + 'lr_f1.xlsx',index=False,header=False)
    nb_f1_df.to_excel(savepath + prefix + 'nb_f1.xlsx',index=False,header=False)
    rf_max_f1_df.to_excel(savepath + prefix + 'rf_max_f1.xlsx',index=False,header=False)
    rf_sqrt_f1_df.to_excel(savepath + prefix + 'rf_sqrt_f1.xlsx',index=False,header=False)
    rf_log2_f1_df.to_excel(savepath + prefix + 'rf_log2_f1.xlsx',index=False,header=False)
    svm_f1_df.to_excel(savepath + prefix + 'svm_f1.xlsx',index=False,header=False)
    svm_C_f1_df.to_excel(savepath + prefix + 'svm_C_f1.xlsx',index=False,header=False)
    svm_rbf_f1_df.to_excel(savepath + prefix + 'svm_rbf_f1.xlsx',index=False,header=False)
    svm_rbf_C_f1_df.to_excel(savepath + prefix + 'svm_rbf_C_f1.xlsx',index=False,header=False)
    svm_rbf_gamma_f1_df.to_excel(savepath + prefix + 'svm_rbf_gamma_f1.xlsx',index=False,header=False)


    mlp_auc_df = pd.DataFrame(mlp_auc[condition_count,:,:])
    adab_auc_df = pd.DataFrame(adab_auc[condition_count,:,:])
    knn_auc_df = pd.DataFrame(knn_auc[condition_count,:,:])
    knn_k_auc_df = pd.DataFrame(knn_k_auc[condition_count,:,:])
    lr_auc_df = pd.DataFrame(lr_auc[condition_count,:,:])
    nb_auc_df = pd.DataFrame(nb_auc[condition_count,:,:])
    rf_max_auc_df = pd.DataFrame(rf_max_auc[condition_count,:,:])
    rf_sqrt_auc_df = pd.DataFrame(rf_sqrt_auc[condition_count,:,:])
    rf_log2_auc_df = pd.DataFrame(rf_log2_auc[condition_count,:,:])
    rf_max_auc_df = pd.DataFrame(rf_max_auc[condition_count,:,:])
    svm_auc_df = pd.DataFrame(svm_auc[condition_count,:,:])
    svm_C_auc_df = pd.DataFrame(svm_C_auc[condition_count,:,:])
    svm_rbf_auc_df = pd.DataFrame(svm_rbf_auc[condition_count,:,:])
    svm_rbf_C_auc_df = pd.DataFrame(svm_rbf_C_auc[condition_count,:,:])
    svm_rbf_gamma_auc_df = pd.DataFrame(svm_rbf_gamma_auc[condition_count,:,:])



    # ranked_feature_auc_df_tranposed.to_excel(savepath + prefix + 'Ranked_features.xlsx',index=False,header=False)
    mlp_auc_df.to_excel(savepath + prefix + 'mlp_auc.xlsx',index=False,header=False)
    adab_auc_df.to_excel(savepath + prefix + 'adab_auc.xlsx',index=False,header=False)
    knn_auc_df.to_excel(savepath + prefix + 'kNN_auc.xlsx',index=False,header=False)
    knn_k_auc_df.to_excel(savepath + prefix + 'kNN_k_auc.xlsx',index=False,header=False)
    lr_auc_df.to_excel(savepath + prefix + 'lr_auc.xlsx',index=False,header=False)
    nb_auc_df.to_excel(savepath + prefix + 'nb_auc.xlsx',index=False,header=False)
    rf_max_auc_df.to_excel(savepath + prefix + 'rf_max_auc.xlsx',index=False,header=False)
    rf_sqrt_auc_df.to_excel(savepath + prefix + 'rf_sqrt_auc.xlsx',index=False,header=False)
    rf_log2_auc_df.to_excel(savepath + prefix + 'rf_log2_auc.xlsx',index=False,header=False)
    svm_auc_df.to_excel(savepath + prefix + 'svm_auc.xlsx',index=False,header=False)
    svm_C_auc_df.to_excel(savepath + prefix + 'svm_C_auc.xlsx',index=False,header=False)
    svm_rbf_auc_df.to_excel(savepath + prefix + 'svm_rbf_auc.xlsx',index=False,header=False)
    svm_rbf_C_auc_df.to_excel(savepath + prefix + 'svm_rbf_C_auc.xlsx',index=False,header=False)
    svm_rbf_gamma_auc_df.to_excel(savepath + prefix + 'svm_rbf_gamma_auc.xlsx',index=False,header=False)




    mlp_max_feature=np.int32(mlp[condition_count,trial+1,N_features+1])
    

    test = SelectKBest(k=mlp_max_feature)
    fit = test.fit(X, y)
    selected_X = fit.transform(X)

    model = MLPClassifier(max_iter=10000,hidden_layer_sizes=[np.int32(np.floor((mlp_max_feature+5)/2)),np.int32(np.floor((mlp_max_feature+5)/2))])

    model.fit(selected_X,y)    

    model_filename = savepath_model + 'mlp_' + str(mlp_max_feature) +  ".joblib"
    joblib.dump(model, model_filename)




    adab_max_feature=np.int32(adab[condition_count,trial+1,N_features+1])


    test = SelectKBest(k=adab_max_feature)
    fit = test.fit(X, y)
    selected_X = fit.transform(X)

    model = AdaBoostClassifier(n_estimators=100)

    model.fit(selected_X,y)    

    model_filename = savepath_model + 'adab_' + str(adab_max_feature) +  ".joblib"
    joblib.dump(model, model_filename)




    lr_max_feature=np.int32(lr[condition_count,trial+1,N_features+1])


    test = SelectKBest(k=lr_max_feature)
    fit = test.fit(X, y)
    selected_X = fit.transform(X)

    model = LogisticRegression()

    model.fit(selected_X,y)    

    model_filename = savepath_model + 'lr_' + str(lr_max_feature) +  ".joblib"
    joblib.dump(model, model_filename)





    nb_max_feature=np.int32(nb[condition_count,trial+1,N_features+1])


    test = SelectKBest(k=nb_max_feature)
    fit = test.fit(X, y)
    selected_X = fit.transform(X)

    model = GaussianNB()

    model.fit(selected_X,y)    

    model_filename = savepath_model + 'nb_' + str(nb_max_feature) +  ".joblib"
    joblib.dump(model, model_filename)





    rf_max_max_feature=np.int32(rf_max[condition_count,trial+1,N_features+1])


    test = SelectKBest(k=rf_max_max_feature)
    fit = test.fit(X, y)
    selected_X = fit.transform(X)

    model = RandomForestClassifier(n_estimators=100,max_features=None)

    model.fit(selected_X,y)    

    model_filename = savepath_model + 'rf_max_' + str(rf_max_max_feature) +  ".joblib"
    joblib.dump(model, model_filename)




    rf_sqrt_max_feature=np.int32(rf_sqrt[condition_count,trial+1,N_features+1])


    test = SelectKBest(k=rf_sqrt_max_feature)
    fit = test.fit(X, y)
    selected_X = fit.transform(X)

    model = RandomForestClassifier(n_estimators=100,max_features='sqrt')

    model.fit(selected_X,y)    

    model_filename = savepath_model + 'rf_sqrt_' + str(rf_sqrt_max_feature) +  ".joblib"
    joblib.dump(model, model_filename)
    



    rf_log2_max_feature=np.int32(rf_log2[condition_count,trial+1,N_features+1])


    test = SelectKBest(k=rf_log2_max_feature)
    fit = test.fit(X, y)
    selected_X = fit.transform(X)

    model = RandomForestClassifier(n_estimators=100,max_features='log2')

    model.fit(selected_X,y)    

    model_filename = savepath_model + 'rf_log2_' + str(rf_log2_max_feature) +  ".joblib"
    joblib.dump(model, model_filename)



    knn_max_feature=np.int32(knn[condition_count,trial+1,N_features+1])

    temp_knn_alltrial=knn[condition_count,1:trial+1,knn_max_feature-1]

    temp_knn_alltrial_max=max(temp_knn_alltrial)

    temp_knn_max_trial=np.where(temp_knn_alltrial==temp_knn_alltrial_max)

    knn_max_trial = temp_knn_max_trial[0][0]+1

    # print(knn_max_trial)

    knn_max_k = np.int32(knn_k[condition_count,knn_max_trial,knn_max_feature-1])

    # print(knn_max_k)
    test = SelectKBest(k=knn_max_feature)
    fit = test.fit(X, y)
    selected_X = fit.transform(X)

    model = KNeighborsClassifier(n_neighbors=knn_max_k)

    model.fit(selected_X,y)    

    model_filename = savepath_model + 'knn_' + str(knn_max_feature) +  ".joblib"
    joblib.dump(model, model_filename)




    svm_max_feature=np.int32(svm_[condition_count,trial+1,N_features+1])

    temp_svm_alltrial=svm_[condition_count,1:trial+1,svm_max_feature-1]

    temp_svm_alltrial_max=max(temp_svm_alltrial)

    temp_svm_max_trial=np.where(temp_svm_alltrial==temp_svm_alltrial_max)

    svm_max_trial = temp_svm_max_trial[0][0]+1

    # print(svm_max_trial)

    svm_max_C = svm_C[condition_count,svm_max_trial,svm_max_feature-1]


    test = SelectKBest(k=svm_max_feature)
    fit = test.fit(X, y)
    selected_X = fit.transform(X)

    model = svm.SVC(kernel='linear',C=svm_max_C)

    model.fit(selected_X,y)    

    model_filename = savepath_model + 'svm_' + str(svm_max_feature) +  ".joblib"
    joblib.dump(model, model_filename)




    svm_rbf_max_feature=np.int32(svm_rbf[condition_count,trial+1,N_features+1])

    temp_svm_rbf_alltrial=svm_rbf[condition_count,1:trial+1,svm_rbf_max_feature-1]

    temp_svm_rbf_alltrial_max=max(temp_svm_rbf_alltrial)

    temp_svm_rbf_max_trial=np.where(temp_svm_rbf_alltrial==temp_svm_rbf_alltrial_max)

    svm_rbf_max_trial = temp_svm_rbf_max_trial[0][0]+1

    # print(svm_rbf_max_trial)

    svm_rbf_max_C = svm_rbf_C[condition_count,svm_rbf_max_trial,svm_rbf_max_feature-1]
    svm_rbf_max_gamma = svm_rbf_gamma[condition_count,svm_rbf_max_trial,svm_rbf_max_feature-1]
    
    test = SelectKBest(k=svm_rbf_max_feature)
    fit = test.fit(X, y)
    selected_X = fit.transform(X)

    model = svm.SVC(kernel='rbf',C=svm_rbf_max_C,gamma=svm_rbf_max_gamma)

    model.fit(selected_X,y)    

    model_filename = savepath_model + 'svm_rbf_' + str(svm_rbf_max_feature) +  ".joblib"
    joblib.dump(model, model_filename)

alpha=0.05

# t_stat,p_value=ttest_ind(mlp[0,1:trial+1,temp_mlp].ravel(),mlp[1,1:trial+1,temp_mlp_2].ravel())
# mlp_p=p_value
# mlp_t=t_stat
# mlp_decision=(p_value/2 < alpha and t_stat>0)
# mlp_control_acc=np.mean(mlp[0,1:trial+1,temp_mlp].ravel())
# mlp_control_std=np.std(mlp[0,1:trial+1,temp_mlp].ravel())
mlp_baseline_acc=np.mean(mlp[0,1:trial+1,temp_mlp].ravel())
mlp_baseline_std=np.std(mlp[0,1:trial+1,temp_mlp].ravel())

# t_stat,p_value=ttest_ind(adab[0,1:trial+1,temp_adab].ravel(),adab[1,1:trial+1,temp_adab_2].ravel())
# adab_p=p_value
# adab_t=t_stat
# adab_decision=(p_value/2 < alpha and t_stat>0)
# adab_control_acc=np.mean(adab[0,1:trial+1,temp_adab].ravel())
# adab_control_std=np.std(adab[0,1:trial+1,temp_adab].ravel())
adab_baseline_acc=np.mean(adab[0,1:trial+1,temp_adab].ravel())
adab_baseline_std=np.std(adab[0,1:trial+1,temp_adab].ravel())

# t_stat,p_value=ttest_ind(knn[0,1:trial+1,temp_knn].ravel(),knn[1,1:trial+1,temp_knn_2].ravel())
# knn_p=p_value
# knn_t=t_stat
# knn_decision=(p_value/2 < alpha and t_stat>0)
# knn_control_acc=np.mean(knn[0,1:trial+1,temp_knn].ravel())
# knn_control_std=np.std(knn[0,1:trial+1,temp_knn].ravel())
knn_baseline_acc=np.mean(knn[0,1:trial+1,temp_knn].ravel())
knn_baseline_std=np.std(knn[0,1:trial+1,temp_knn].ravel())

# t_stat,p_value=ttest_ind(lr[0,1:trial+1,temp_lr].ravel(),lr[1,1:trial+1,temp_lr_2].ravel())
# lr_p=p_value
# lr_t=t_stat
# lr_decision=(p_value/2 < alpha and t_stat>0)
# lr_control_acc=np.mean(lr[0,1:trial+1,temp_lr].ravel())
# lr_control_std=np.std(lr[0,1:trial+1,temp_lr].ravel())
lr_baseline_acc=np.mean(lr[0,1:trial+1,temp_lr].ravel())
lr_baseline_std=np.std(lr[0,1:trial+1,temp_lr].ravel())

# t_stat,p_value=ttest_ind(nb[0,1:trial+1,temp_nb].ravel(),nb[1,1:trial+1,temp_nb_2].ravel())
# nb_p=p_value
# nb_t=t_stat
# nb_decision=(p_value/2 < alpha and t_stat>0)
# nb_control_acc=np.mean(nb[0,1:trial+1,temp_nb].ravel())
# nb_control_std=np.std(nb[0,1:trial+1,temp_nb].ravel())
nb_baseline_acc=np.mean(nb[0,1:trial+1,temp_nb].ravel())
nb_baseline_std=np.std(nb[0,1:trial+1,temp_nb].ravel())

# t_stat,p_value=ttest_ind(rf_max[0,1:trial+1,temp_rf_max].ravel(),rf_max[1,1:trial+1,temp_rf_max_2].ravel())
# rf_max_p=p_value
# rf_max_t=t_stat
# rf_max_decision=(p_value/2 < alpha and t_stat>0)
# rf_max_control_acc=np.mean(rf_max[0,1:trial+1,temp_rf_max].ravel())
# rf_max_control_std=np.std(rf_max[0,1:trial+1,temp_rf_max].ravel())
rf_max_baseline_acc=np.mean(rf_max[0,1:trial+1,temp_rf_max].ravel())
rf_max_baseline_std=np.std(rf_max[0,1:trial+1,temp_rf_max].ravel())

# t_stat,p_value=ttest_ind(rf_log2[0,1:trial+1,temp_rf_log2].ravel(),rf_log2[1,1:trial+1,temp_rf_log2_2].ravel())
# rf_log2_p=p_value
# rf_log2_t=t_stat
# rf_log2_decision=(p_value/2 < alpha and t_stat>0)
# rf_log2_control_acc=np.mean(rf_log2[0,1:trial+1,temp_rf_log2].ravel())
# rf_log2_control_std=np.std(rf_log2[0,1:trial+1,temp_rf_log2].ravel())
rf_log2_baseline_acc=np.mean(rf_log2[0,1:trial+1,temp_rf_log2].ravel())
rf_log2_baseline_std=np.std(rf_log2[0,1:trial+1,temp_rf_log2].ravel())

# t_stat,p_value=ttest_ind(rf_sqrt[0,1:trial+1,temp_rf_sqrt].ravel(),rf_sqrt[1,1:trial+1,temp_rf_sqrt_2].ravel())
# rf_sqrt_p=p_value
# rf_sqrt_t=t_stat
# rf_sqrt_decision=(p_value/2 < alpha and t_stat>0)
# rf_sqrt_control_acc=np.mean(rf_sqrt[0,1:trial+1,temp_rf_sqrt].ravel())
# rf_sqrt_control_std=np.std(rf_sqrt[0,1:trial+1,temp_rf_sqrt].ravel())
rf_sqrt_baseline_acc=np.mean(rf_sqrt[0,1:trial+1,temp_rf_sqrt].ravel())
rf_sqrt_baseline_std=np.std(rf_sqrt[0,1:trial+1,temp_rf_sqrt].ravel())

# t_stat,p_value=ttest_ind(svm_[0,1:trial+1,temp_svm_].ravel(),svm_[1,1:trial+1,temp_svm_2].ravel())
# svm_p=p_value
# svm_t=t_stat
# svm_decision=(p_value/2 < alpha and t_stat>0)
# svm_control_acc=np.mean(svm_[0,1:trial+1,temp_svm_].ravel())
# svm_control_std=np.std(svm_[0,1:trial+1,temp_svm_].ravel())
svm_baseline_acc=np.mean(svm_[0,1:trial+1,temp_svm_].ravel())
svm_baseline_std=np.std(svm_[0,1:trial+1,temp_svm_].ravel())

# t_stat,p_value=ttest_ind(svm_rbf[0,1:trial+1,temp_svm_rbf].ravel(),svm_rbf[1,1:trial+1,temp_svm_rbf_2].ravel())
# svm_rbf_p=p_value
# svm_rbf_t=t_stat
# svm_rbf_decision=(p_value/2 < alpha and t_stat>0)
# svm_rbf_control_acc=np.mean(svm_rbf[0,1:trial+1,temp_svm_rbf].ravel())
# svm_rbf_control_std=np.std(svm_rbf[0,1:trial+1,temp_svm_rbf].ravel())
svm_rbf_baseline_acc=np.mean(svm_rbf[0,1:trial+1,temp_svm_rbf].ravel())
svm_rbf_baseline_std=np.std(svm_rbf[0,1:trial+1,temp_svm_rbf].ravel())	

d = {'mlp': [mlp_baseline_acc, mlp_baseline_std]
     , 'adab': [adab_baseline_acc, adab_baseline_std]
     , 'knn': [knn_baseline_acc, knn_baseline_std]
     , 'lr': [lr_baseline_acc, lr_baseline_std]
     , 'nb': [nb_baseline_acc, nb_baseline_std]
     , 'rf_max': [rf_max_baseline_acc, rf_max_baseline_std]
     , 'rf_log2': [rf_log2_baseline_acc, rf_log2_baseline_std]
     , 'rf_sqrt': [rf_sqrt_baseline_acc, rf_sqrt_baseline_std]
     , 'svm': [svm_baseline_acc, svm_baseline_std]
     , 'svm_rbf': [svm_rbf_baseline_acc, svm_rbf_baseline_std]}
rows=['control-exp acc.','control-exp std.']
df = pd.DataFrame(data=d,index=rows)

df.to_excel(savepath + 'mixed features_SUMMARY level ' + level + ' accuracy.xlsx')




# t_stat,p_value=ttest_ind(mlp[1,1:trial+1,temp_mlp_2].ravel(),mlp[0,1:trial+1,temp_mlp].ravel())
# mlp_p=p_value
# mlp_t=t_stat
# mlp_decision=(p_value/2 < alpha and t_stat>0)

# t_stat,p_value=ttest_ind(adab[1,1:trial+1,temp_adab_2].ravel(),adab[0,1:trial+1,temp_adab].ravel())
# adab_p=p_value
# adab_t=t_stat
# adab_decision=(p_value/2 < alpha and t_stat>0)

# t_stat,p_value=ttest_ind(knn[1,1:trial+1,temp_knn_2].ravel(),knn[0,1:trial+1,temp_knn].ravel())
# knn_p=p_value
# knn_t=t_stat
# knn_decision=(p_value/2 < alpha and t_stat>0)

# t_stat,p_value=ttest_ind(lr[1,1:trial+1,temp_lr_2].ravel(),lr[0,1:trial+1,temp_lr].ravel())
# lr_p=p_value
# lr_t=t_stat
# lr_decision=(p_value/2 < alpha and t_stat>0)

# t_stat,p_value=ttest_ind(nb[1,1:trial+1,temp_nb_2].ravel(),nb[0,1:trial+1,temp_nb].ravel())
# nb_p=p_value
# nb_t=t_stat
# nb_decision=(p_value/2 < alpha and t_stat>0)

# t_stat,p_value=ttest_ind(rf_max[1,1:trial+1,temp_rf_max_2].ravel(),rf_max[0,1:trial+1,temp_rf_max].ravel())
# rf_max_p=p_value
# rf_max_t=t_stat
# rf_max_decision=(p_value/2 < alpha and t_stat>0)

# t_stat,p_value=ttest_ind(rf_log2[1,1:trial+1,temp_rf_log2_2].ravel(),rf_log2[0,1:trial+1,temp_rf_log2].ravel())
# rf_log2_p=p_value
# rf_log2_t=t_stat
# rf_log2_decision=(p_value/2 < alpha and t_stat>0)

# t_stat,p_value=ttest_ind(rf_sqrt[1,1:trial+1,temp_rf_sqrt_2].ravel(),rf_sqrt[0,1:trial+1,temp_rf_sqrt].ravel())
# rf_sqrt_p=p_value
# rf_sqrt_t=t_stat
# rf_sqrt_decision=(p_value/2 < alpha and t_stat>0)

# t_stat,p_value=ttest_ind(svm_[1,1:trial+1,temp_svm_2].ravel(),svm_[0,1:trial+1,temp_svm_].ravel())
# svm_p=p_value
# svm_t=t_stat
# svm_decision=(p_value/2 < alpha and t_stat>0)

# t_stat,p_value=ttest_ind(svm_rbf[1,1:trial+1,temp_svm_rbf_2].ravel(),svm_rbf[0,1:trial+1,temp_svm_rbf].ravel())
# svm_rbf_p=p_value
# svm_rbf_t=t_stat
# svm_rbf_decision=(p_value/2 < alpha and t_stat>0)

# d = {'mlp': [mlp_p, mlp_t, mlp_decision, mlp_control_acc, mlp_control_std, mlp_baseline_acc, mlp_baseline_std]
#      , 'adab': [adab_p, adab_t, adab_decision, adab_control_acc, adab_control_std, adab_baseline_acc, adab_baseline_std]
#      , 'knn': [knn_p, knn_t, knn_decision, knn_control_acc, knn_control_std, knn_baseline_acc, knn_baseline_std]
#      , 'lr': [lr_p, lr_t, lr_decision, lr_control_acc, lr_control_std, lr_baseline_acc, lr_baseline_std]
#      , 'nb': [nb_p, nb_t, nb_decision, nb_control_acc, nb_control_std, nb_baseline_acc, nb_baseline_std]
#      , 'rf_max': [rf_max_p, rf_max_t, rf_max_decision, rf_max_control_acc, rf_max_control_std, rf_max_baseline_acc, rf_max_baseline_std]
#      , 'rf_log2': [rf_log2_p, rf_log2_t, rf_log2_decision, rf_log2_control_acc, rf_log2_control_std, rf_log2_baseline_acc, rf_log2_baseline_std]
#      , 'rf_sqrt': [rf_sqrt_p, rf_sqrt_t, rf_sqrt_decision, rf_sqrt_control_acc, rf_sqrt_control_std, rf_sqrt_baseline_acc, rf_sqrt_baseline_std]
#      , 'svm': [svm_p, svm_t, svm_decision, svm_control_acc, svm_control_std, svm_baseline_acc, svm_baseline_std]
#      , 'svm_rbf': [svm_rbf_p, svm_rbf_t, svm_rbf_decision, svm_rbf_control_acc, svm_rbf_control_std, svm_rbf_baseline_acc, svm_rbf_baseline_std]}
# rows=['p-Value','t','Decision','control acc.','control std','baseline acc.','baseline std.']
# df = pd.DataFrame(data=d,index=rows)

# df.to_excel(savepath + 'mixed features_SUMMARY level ' + level + ' baseline_higher_control.xlsx')