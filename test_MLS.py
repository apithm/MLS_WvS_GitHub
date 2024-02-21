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

import openpyxl
import pandas as pd
from itertools import combinations
from openpyxl import load_workbook, Workbook
from pathlib import Path

classifier_list=[1,2,3,4,5,6,7,8,9,10,11,12,13,14,15,16,17,18,19,20]
# N_classifier_chosen_list=[1,2,3,4,5,6,7,8,9,10,11,12,13,14,15,16,17,18,19,20]
# N_classifier_chosen_list=[12]
# N_classifier_chosen_list=[1,2,3,4,5,6,7] #ws
# N_classifier_chosen_list=[1,2,3,4,5,6] #ws
# N_classifier_chosen_list=[7] #ws
# N_classifier_chosen_list=[15,16,17,18,19,20] #ws
# N_classifier_chosen_list=[20]
N_classifier_chosen_list=[13]


class_tobetested=0
# class_tobetested=1
# class_tobetested=2
# class_tobetested=3
# class_tobetested=4

gender='male'
# gender='female_1'
# gender='female_2'




path_save = 'D:\\Dropbox\\NECTEC\\My Documents\\StressClassification_NRIIS\\Result\\All_cls_4class_1vs1_OVR_' + gender + '_gender\\'
if os.path.isdir(path_save)==False:
        os.makedirs(path_save)

# fullname_result=path_save+'1-7 Class ' + str(class_tobetested) + '.xlsx'
# fullname_result=path_save+'13-20 Class ' + str(class_tobetested) + '.xlsx'
if len(N_classifier_chosen_list)>1:
    fullname_result=path_save + str(N_classifier_chosen_list[0]) + '-' + str(N_classifier_chosen_list[len(N_classifier_chosen_list)-1])  + ' Class ' + str(class_tobetested) + '.xlsx'
else:
    fullname_result=path_save + 'Only ' + str(N_classifier_chosen_list[0]) + ' Class ' + str(class_tobetested) + '.xlsx'



b=0
for i in range(len(N_classifier_chosen_list)):
    comb = combinations(classifier_list, N_classifier_chosen_list[i])
    for c in list(comb):
        b=b+1
    # a=0

# print(b)
trial=50
# trial=3

N_feature=138
# N_feature=20



# if class_tobetested==0:
#     b=b*4

path = 'D:\\Dropbox\\NECTEC\\My Documents\\StressClassification_NRIIS\\Result\\Full_ECG_EEG_4class_' + gender + '_gender\\'
path_2 = 'D:\\Dropbox\\NECTEC\\My Documents\\StressClassification_NRIIS\\Result\\Full_ECG_EEG_4class_OVR_' + gender + '_gender\\'

total_class=5

condition='baseline'

df_combine=pd.DataFrame
k=0
for classifier_chosen_count_list_count in range(len(N_classifier_chosen_list)):
    classifier_chosen=N_classifier_chosen_list[classifier_chosen_count_list_count]

    comb = combinations(classifier_list, classifier_chosen)

    N_comb=0
    for comb_count in list(comb):
        N_comb=N_comb+1

    comb = combinations(classifier_list, classifier_chosen)



    overall_set_accuracy=np.zeros([total_class,N_comb])
    overall_set_mcc=np.zeros([total_class,N_comb])
    overall_set_sensitivity=np.zeros([total_class,N_comb])
    overall_set_specificity=np.zeros([total_class,N_comb])
    overall_set_tp=np.zeros([total_class,N_comb])
    overall_set_tn=np.zeros([total_class,N_comb])
    overall_set_fp=np.zeros([total_class,N_comb])
    overall_set_fn=np.zeros([total_class,N_comb])
    overall_set_f1_score=np.zeros([total_class,N_comb])
    overall_set_auc=np.zeros([total_class,N_comb])
    overall_set_accuracy_diag=np.zeros([N_comb,1])
    overall_set_accuracy_nondiag=np.zeros([N_comb,1])



    # for class_tobetest_count in range(1,total_class+1):
    # class_tobetest_count=class_tobetested
    comb_count_2=0
    comb = combinations(classifier_list, classifier_chosen)
    for comb_count in list(comb):
        
        

        comb_count_2=comb_count_2+1

        dataset_test = 'ECG_EEG_' + gender + '_baseline_exp_OVR_' + str(class_tobetested) + 'vsRest_test'


        filenam_test = dataset_test + '.csv'
        path_test_file = 'D:\\Dropbox\\NECTEC\\My Documents\\StressClassification_NRIIS\\ECG & EEG_2\\Gender\\'
        fullname_test_file = path_test_file + filenam_test
        data = pd.read_csv(fullname_test_file,header=0)

        subject= len(data.index)
        # subject=9

    

    
        print(N_classifier_chosen_list)
        model_name=list()
        for i in range(len(comb_count)):
            if comb_count[i]%10 == 1:
                model_name.append('adab')
            if comb_count[i]%10 == 2:
                model_name.append('knn')
            if comb_count[i]%10 == 3:
                model_name.append('lr')   
            if comb_count[i]%10 == 4:
                model_name.append('mlp')
            if comb_count[i]%10 == 5:
                model_name.append('nb')    
            if comb_count[i]%10 == 6:
                model_name.append('rf_log2')
            if comb_count[i]%10 == 7:
                model_name.append('rf_max')
            if comb_count[i]%10 == 8:
                model_name.append('rf_sqrt')
            if comb_count[i]%10 == 9:
                model_name.append('svm')
            if comb_count[i]%10 == 10:
                model_name.append('svm_rbf')
        

        N_sub_model=len(model_name)
        
        # for set_model_for_class_count in range(1,total_class+1):
        # set_model_for_class_count = class_tobetested
        # set_prediction=np.zeros([subject,1])
        # set_prediction=
        set_correct=0
        tp=0
        tn=0
        fp=0
        fn=0

        # print('Testing class: ' + str(class_tobetest_count) + ', Set model: ' + str(set_model_for_class_count))

        # for subj_count in range(1,subject+1):

        sub_model_prediction=np.zeros([N_sub_model,subject])

        # print('Testing class: ' + str(class_tobetest_count) + ', Set model: ' + str(set_model_for_class_count) + ' ' +  str(subj_count))
        if class_tobetested>0:
            for sub_model_count in range(1,N_sub_model+1):
                
                # print(k)
                
            



                filename='mixed features_' + condition +  '_class_' + str(class_tobetested)+  '_' + model_name[sub_model_count-1] + '.xlsx'

                

                if comb_count[sub_model_count-1]<11:
                    fullname = path + filename
                    path_model = path + '\\model\\' + condition + '\\' + str(class_tobetested) + '\\'
                else:
                    fullname = path_2 + filename
                    path_model = path_2 + '\\model\\' + condition + '\\' + str(class_tobetested) + '\\'
                
                df1 = pd.read_excel(fullname)
                max_feature=np.int32(df1.loc[trial].iat[N_feature+1])

                fullname_model = path_model + model_name[sub_model_count-1] + '_' + str(max_feature) + '.joblib'
                
                # print(fullname_model)
                a=0
                # # print(feature)

                model=joblib.load(fullname_model)


                if comb_count[sub_model_count-1]<11:
                    acc_filename = path + "mixed features_" + condition + "_class_" + str(class_tobetested) + "_" + model_name[sub_model_count-1] + ".xlsx"
                    
                    feature_filename = path + "mixed features_" + condition + "_class_" + str(class_tobetested) + "_Ranked_features.xlsx"
                else:
                    acc_filename = path_2 + "mixed features_" + condition + "_class_" + str(class_tobetested) + "_" + model_name[sub_model_count-1] + ".xlsx"
                    
                    feature_filename = path_2 + "mixed features_" + condition + "_class_" + str(class_tobetested) + "_Ranked_features.xlsx"

                df_feature = pd.read_excel(feature_filename,header=None)

                selected_feature = df_feature.iloc[0:max_feature]
                selected_feature_list=list()
                for i in range(0,max_feature):
                    selected_feature_list.append(np.int32(selected_feature[0][i])-1)

                selected_feature_list_sorted=list()
                selected_feature_list_sorted=sorted(selected_feature_list)



                data_selected=data.iloc[:,selected_feature_list_sorted]

                X = data_selected.values
                y_temp = data['Class'].values

                y=y_temp

                a=0


                # X = X.reshape(1,-1)
                X_size=X.shape
                # print(X_size[1])
                if X_size[0]==1 & X_size[1]==1:
                    X = X.reshape(1,-1)
                if X_size[1]>1:
                    StdScaler = preprocessing.StandardScaler()    
                    X = StdScaler.fit_transform(X)
                else:
                    a=0

                sub_model_prediction[sub_model_count-1,:]=model.predict(X)
                a=0
        else:
            
            for sub_model_count in range(1,N_sub_model+1):
                # k=k+1
                # print(k)
                # progress = k/b *100
                # print(str(class_tobetested) + ': ' + str(progress) + ' %')
            
                if comb_count[sub_model_count-1]<11:
                    total_class=5
                    temp_predict=np.zeros([subject,total_class-1])
                else:
                    total_class=1+1
                    temp_predict=np.zeros([subject,1])
                
                for sub_model_count_2 in range(1,total_class):

                    filename='mixed features_' + condition +  '_class_' + str(sub_model_count_2)+  '_' + model_name[sub_model_count-1] + '.xlsx'


                    if comb_count[sub_model_count-1]<11:
                        fullname = path + filename
                        path_model = path + '\\model\\' + condition + '\\' + str(sub_model_count_2) + '\\'
                    else:
                        fullname = path_2 + filename
                        path_model = path_2 + '\\model\\' + condition + '\\' + str(sub_model_count_2) + '\\'
                    
                    df1 = pd.read_excel(fullname)
                    max_feature=np.int32(df1.loc[trial].iat[N_feature+1])

                    fullname_model = path_model + model_name[sub_model_count-1] + '_' + str(max_feature) + '.joblib'
                    
                    # print(fullname_model)
                    a=0
                    # # print(feature)

                    model=joblib.load(fullname_model)


                    if comb_count[sub_model_count-1]<11:
                        acc_filename = path + "mixed features_" + condition + "_class_" + str(sub_model_count_2) + "_" + model_name[sub_model_count-1] + ".xlsx"
                        
                        feature_filename = path + "mixed features_" + condition + "_class_" + str(sub_model_count_2) + "_Ranked_features.xlsx"
                    else:
                        acc_filename = path_2 + "mixed features_" + condition + "_class_" + str(sub_model_count_2) + "_" + model_name[sub_model_count-1] + ".xlsx"
                        
                        feature_filename = path_2 + "mixed features_" + condition + "_class_" + str(sub_model_count_2) + "_Ranked_features.xlsx"

                    df_feature = pd.read_excel(feature_filename,header=None)

                    selected_feature = df_feature.iloc[0:max_feature]
                    selected_feature_list=list()
                    for i in range(0,max_feature):
                        selected_feature_list.append(np.int32(selected_feature[0][i])-1)

                    selected_feature_list_sorted=list()
                    selected_feature_list_sorted=sorted(selected_feature_list)



                    data_selected=data.iloc[:,selected_feature_list_sorted]

                    X = data_selected.values
                    y_temp = data['Class'].values

                    y=y_temp

                    a=0


                    # X = X.reshape(1,-1)
                    X_size=X.shape
                    # print(X_size[1])
                    if X_size[0]==1 & X_size[1]==1:
                        X = X.reshape(1,-1)
                    if X_size[1]>1:
                        StdScaler = preprocessing.StandardScaler()    
                        X = StdScaler.fit_transform(X)
                    else:
                        a=0

                    temp_pred=model.predict(X)

                    if comb_count[sub_model_count-1]<11:
                        for p in range(0,len(temp_pred)):
                            if temp_pred[p]==0:
                                temp_pred[p]=1
                            else:
                                temp_pred[p]=0                    

                    temp_predict[:,sub_model_count_2-1]=temp_pred
                    a=0

                    # if comb_count[sub_model_count-1]>10:
                    #     print(temp_predict)




                    
                    # sub_model_prediction[sub_model_count-1,:]=
                    a=0
                # print(temp_predict)
                for i in range(subject):
                    # print((temp_predict[i,:]))
                    # print(len(temp_predict[i,:]) / 2.0)
                    if sum(temp_predict[i,:])  > len(temp_predict[i,:]) / 2.0:
                        temp_set_prediction = 1
                    else:
                        temp_set_prediction= 0

                    sub_model_prediction[sub_model_count-1,i]=temp_set_prediction


        # print(sub_model_prediction)
        for subj_count in range(1,subject+1):
            if sum(sub_model_prediction[:,subj_count-1])  > len(sub_model_prediction[:,subj_count-1]) / 2.0:
                set_prediction = 1
            else:
                set_prediction= 0

            # print(sub_model_prediction[:,subj_count-1])
            # print(y[subj_count-1])

            if set_prediction == 0 and y[subj_count-1]==0:
                tn = tn + 1
            
            elif set_prediction == 0 and y[subj_count-1]==1:
                fn = fn + 1

            elif set_prediction == 1 and y[subj_count-1]==0:
                fp = fp + 1

            elif set_prediction == 1 and y[subj_count-1]==1:
                tp = tp + 1


        accuracy = (tp+tn)/(tp+tn+fn+fp)*100

        mcc=0

        if (tp)>0 and (tn>0) and (fp>0) and (fn>0):
            mcc = ((tp*tn)-(fp*fn))/np.sqrt((tp+fp)*(tp+fn)*(tn+fp)*(tn+fn))
        elif (tp==0) or (tn==0):
            mcc = 0
        elif (fp==0) or (fn==0):
            mcc = 1

        if (tp+fp)>0:
            precision = tp/(tp+fp)
        else:
            precision = 0 
        
        if (tp+fn)>0:
            recall = tp/(tp+fn)
        else:
            recall = 0

        sensitivity = recall * 100

        if (tn+fp)>0:
            specificity = tn / (tn+fp) * 100
        else:
            specificity = 0
            
        if precision >0 or recall>0:
            f1_score = 2*((precision*recall)/(precision+recall))
        else:
            f1_score = 0

        if class_tobetested==3 and class_tobetested==1:
            a=0
        tpr = recall #same as recall, sensitivity
        if (fp+tn)>0:
            fpr = fp/(fp+tn)
        else:
            fpr=0

        # print(set_correct_percent)
        # print('Testing class: ' + str(class_tobetest_count) + ', Set model: ' + str(set_model_for_class_count) + ', Accuracy: ' +  str(accuracy), ' %')

        overall_set_accuracy[class_tobetested,comb_count_2-1]=accuracy
        overall_set_mcc[class_tobetested,comb_count_2-1]=mcc
        overall_set_f1_score[class_tobetested,comb_count_2-1]=f1_score
        overall_set_sensitivity[class_tobetested,comb_count_2-1]=sensitivity
        overall_set_specificity[class_tobetested,comb_count_2-1]=specificity
        overall_set_tp[class_tobetested,comb_count_2-1]=tp
        overall_set_tn[class_tobetested,comb_count_2-1]=tn
        overall_set_fp[class_tobetested,comb_count_2-1]=fp
        overall_set_fn[class_tobetested,comb_count_2-1]=fn
        
        # overall_set_auc[class_tobetest_count-1,set_model_for_class_count-1]=auc
        k=k+1
        progress = k/b *100
        print(gender + ' ' + str(class_tobetested) + ': ' + str(progress) + ' % (',str(classifier_chosen),': ',str(k),'/',str(b),')')



    classifier_name_list=['ADAB','kNN','LR','MLP','NB','RF_log2','RF_max','RF_sqrt','SVM','SVM_RBF',\
                            'ADAB-OVR','kNN-OVR','LR-OVR','MLP-OVR','NB-OVR','RF_log2-OVR','RF_max-OVR','RF_sqrt-OVR','SVM-OVR','SVM_RBF-OVR']
    comb = combinations(classifier_list, classifier_chosen)

    classifier_name_list_chosen=list()
    for i in list(comb):
        temp_2=''
        for j in range(classifier_chosen):
            temp=classifier_name_list[i[j]-1]
            # temp_1=classifier_name_list[i[j]]
            # temp_2=temp+', '+temp_1
            if j==0:
                temp_2=temp
            else:
                temp_2=temp_2+', '+temp
            # print(temp_2)
            if j==classifier_chosen-1:
                classifier_name_list_chosen.append(temp_2)


    # print(classifier_name_list_chosen)
    # temp='Class '+ str(class_tobetest_count)
    columns=['Accuracy','F1 score','MCC','Sensitivity','Specificity','TP','TN','FP','FN']
    df = pd.DataFrame(data=np.transpose([overall_set_accuracy[class_tobetested,:],\
                                        overall_set_f1_score[class_tobetested,:],\
                                        overall_set_mcc[class_tobetested,:],\
                                        overall_set_sensitivity[class_tobetested,:],\
                                        overall_set_specificity[class_tobetested,:],\
                                        overall_set_tp[class_tobetested,:],\
                                        overall_set_tn[class_tobetested,:],\
                                        overall_set_fp[class_tobetested,:],\
                                        overall_set_fn[class_tobetested,:]]),\
                                        index=classifier_name_list_chosen,columns=columns)

    print(df)
    # if df_combine.empty:
    if classifier_chosen_count_list_count==0:
        df_combine=df
        df_combine_2=df_combine # df_combine is still a DataFrame
    elif classifier_chosen_count_list_count==1:
        print(df_combine)
        df_combine=[df_combine,df] # df_combine was still a DataFrame, but it has become a list by using []
        print(df_combine)
        df_combine_2=pd.concat(df_combine)
    else:
        df_combine.append(df) # to concat another list to an existing list, we must use append
        a=0

        df_combine_2=pd.concat(df_combine)

    
    df_combine_2.to_excel(fullname_result)
