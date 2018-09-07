##Stability selection 
##Authors: Cristina Venturini & Yuxin Sun
# coding: utf-8

# In[1]:



#Import functions 
from os.path import isdir
import collections
from collections import Counter, OrderedDict
from scipy.io import loadmat
from random import sample
import numpy as np
from itertools import chain
from sklearn.model_selection import KFold
from sklearn.linear_model import SGDClassifier
from sklearn.metrics import accuracy_score
import csv
import pandas as pd
from sklearn.model_selection import train_test_split

def flatten(x):
    return [val for sublist in x for val in sublist]





def run_svm(n, k, m, data, alpha_range, k_subsample):
    n_sample = data.shape[0]  # number of samples
    n_feature = data.shape[1]  # number of features
    idx_coef = []  # indices of selected features by feature selection algorithm
    idx_select = []  # indices selected for subset
    samples_select = []
    kf = KFold(n_splits=k)
    for counter in range(n):
        if counter % 10 == 0:
            print('current iteration: %d.' % counter)

        # subsample m features
        idx = sample(range(n_feature), m)
        idx_select.append(idx)

        ##split the observations (patients) 50% keeping the proportion disease/healthy as in the big dataset

        data_n = data[:, idx]
        data_new, X_test, label_new, y_test = train_test_split(data_n,label, test_size = 0.5, stratify=label)
        print(label_new)

        
        #data_new = data_n[samples, :]
        #label_new = label[samples]
        acc = run_cross_validation(label, alpha_range, k, kf)
        # re-train model with optimal parameter
        idx_cv = np.argmax(np.mean(acc, axis=0))
        clf = SGDClassifier(alpha=alpha_range[idx_cv], penalty='l1', loss='hinge')
        clf.fit(data_new, label_new)
        #clf.fit(data[:, idx], label)
        coef = np.squeeze(clf.coef_)
        idx_new = list(np.where(coef != 0)[0])
        idx_coef.append([idx[x] for  x in idx_new])  # indices of selected features
    return idx_select, idx_coef 

def run_cross_validation(label, alpha_range, k, kf):  
    acc = np.empty([k, len(alpha_range)])
    for k_counter, (idx_trn_trn, idx_trn_val) in enumerate(kf.split(range(len(label)))):
        print('%d/%d cross validation.' % (k_counter + 1, k))
        data_trn_trn = data[idx_trn_trn, :]  # smaller training set in cross validation
        data_trn_val = data[idx_trn_val, :]  # validation set in cross validation
        label_trn_trn = label[idx_trn_trn]
        label_trn_val = label[idx_trn_val]

        # l1 svm on validation set
        for alpha_counter, alpha in enumerate(alpha_range):
            clf = SGDClassifier(alpha=alpha, penalty='l1', loss='hinge')
            clf.fit(data_trn_trn, label_trn_trn)
            pred = clf.predict(data_trn_val)
            acc[k_counter, alpha_counter] = accuracy_score(label_trn_val, pred)
    return acc


#write results 

def sort_indices(idx_select, idx_coef):
    """sort indices according to the frequencies to be selected"""
    idx_select = Counter(flatten(idx_select))
    idx_dic = Counter(flatten(idx_coef))
    idx_coef = [(k, idx_dic[k] / float(idx_select[k])) for k in idx_select.keys()]
    res = pd.DataFrame(sorted(idx_coef))
    genes = pd.DataFrame(list(data_tot.keys())[2:]) #make it as df - <3 pandas
    genes = genes.rename(columns={0:'genes'}) #rename column "genes"
    print(genes)
    res = res.rename(columns={1:'freq'})
    print(res)
    together = genes.join(res) #concatenate results with genes names 
    res_sorted = together.sort_values(together.columns[2], ascending=False) #sort values by freq 
    return res_sorted


# In[3]:
##CHANGE HERE

if __name__ == '__main__':
    iter_num = 10  # repeated iterations - make sure to have enough iter to include all genes 
    sampled_features = 3  # sampled features in each iteration - 10% of features ###CHANGE HERE#####
    k_subsample = 5  # k in k-fold cross validation - cross-validation to choose parameter for l1 SVM 
    #i = 62 # select 80% of samples ###CHANGE HERE#####

    data_tot = pd.read_csv('data/input_data.csv') #name of input file
    label = data_tot.Class.values #extract Class to use as label 
    label = np.squeeze(label.reshape(8,1)) #n of sample ###CHANGE HERE#####
    
    data = data_tot.loc[:,'A2M':].values  #get rid of "Patient" and "Class" - check which one is your first gene ###CHANGE HERE#####
    alpha_range = np.logspace(-5, 15, 21, base=2)  # parameters used in l1 svm

    idx_select, idx_coef = run_svm(iter_num, k_subsample, sampled_features, data, alpha_range, k_subsample)
    res_sorted = sort_indices(idx_select, idx_coef)

    res_sorted.to_csv('output/out.csv',sep=',') #save results as csv - this file is ready for SVM in R ###CHANGE HERE#####


