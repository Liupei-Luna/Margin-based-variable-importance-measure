# -*- coding: utf-8 -*-
"""
Created on Wed May 16 16:29:42 2018

@author: Dell
"""


import numpy as np
#import sys
import copy
import os

#from sklearn.cross_validation import ShuffleSplit
from sklearn.metrics import r2_score
from collections import defaultdict
#import sklearn

from sklearn.ensemble import forest
from sklearn.tree import DecisionTreeRegressor, DecisionTreeClassifier, _tree
reload(forest)
from sklearn.datasets import load_breast_cancer, load_iris
from collections import Counter
from matplotlib import pyplot as plt


def split_data(dataset):
    feature = []
    label = []
    
    for i in range(len(dataset)):
        feature_tmp = []
        for k in dataset[i][0:-1]:
            feature_tmp.append(float(k))
        feature.append(feature_tmp)
        label.append(dataset[i][-1])
    feature = np.array(feature)
#    label = np.array(map(int,label))
    return feature,label

class importanceEstimator():
    '''
        Here be documentation.
    '''
    def __init__(self,clf,X,Y,algorithm='gini'):
        clf.fit(X,Y)
        
        self.clf= clf
        if algorithm in ['gini','permutation','mcm','mcm_']:
            self.algorithm= algorithm
        else:
            raise ValueError('Error: algorithm not recognised: '+str(algorithm))


    def fit(self,X,y):
        '''
            This function actually calculates the importances 
            using cross validation.
            Usage:
              imp = fit(X,y)
            Arguments:
              X: feature vector, numpy array
              y: label vector, numpy array
            Return values:
              imp: feature importance vector
        '''
#        scores = defaultdict(list) # Any unknown element is automatically a list
        rf= copy.deepcopy(self.clf)
        #
        #crossvalidate the scores on a number of different random splits of the data
#        
#        for train_idx, test_idx in ShuffleSplit(len(X), self.nCV,):
#            X_train, X_test = X[train_idx], X[test_idx]
#            y_train, y_test = y[train_idx], y[test_idx]
#            rf.fit(X_train, y_train)
            # Get accuracy metr
#            outAcc += rf.oob_score

            
            
        if self.algorithm == 'gini':
            importances = self.giniImportance(rf,X,y) 
        elif self.algorithm == 'permutation':
            importances = self.permutationImportance(rf,X,y) 
        elif self.algorithm == 'mcm':
            importances = self.mcmImportance(rf,X,y)
        elif self.algorithm == 'mcm_':
            importances = self.mcmImportance_(rf,X,y)
        
        # Return mean importance and metric
#        importances= np.array([np.mean(scores[i]) for i in range(X.shape[1])])
        return importances
            
            

    def giniImportance(self,rf,X,y):
        return rf.feature_importances_

    def permutationImportance(self,rf,X,y):
        # Get feature importances
        acc = r2_score(y, rf.predict(X))
        scores= defaultdict(list)
        for i in range(X.shape[1]):
            X_t = X.copy()
            np.random.shuffle(X_t[:, i])
            shuff_acc = r2_score(y, rf.predict(X_t))
            scores[i].append((acc-shuff_acc)/acc)
        return np.array([ np.mean(scores[i]) for i in range(X.shape[1]) ])
    
    
    
    
    def mcmImportance(self,rf,X,y):
      
    #MCM+RFE选择特征
        
#        sample = [[0 for col in xrange(X.shape[1])] for row in xrange(X.shape[0])]
        sample = np.zeros ( X.shape )     
        for tree in rf.estimators_:
            Features = tree.tree_.feature
            for case in tree.oob_ind:
                feaCounter = Counter(Features)
                feaCounter.pop(-2)
                if tree.predict(X[case,:].reshape(1,-1)) == float(y[case]):
                    for k,w in feaCounter.items():
                        sample[case,k] += w
                else:
                    for k,w in feaCounter.items():
                        sample[case,k] -= w
        mea = np.mean(sample,axis=0)/rf.n_estimators
#        print(sample,mea)
        return mea
    def mcmImportance_(self,rf,X,y):
      
    #MCM+RFE选择特征
        
#        sample = [[0 for col in xrange(X.shape[1])] for row in xrange(X.shape[0])]
        sample = np.zeros ( X.shape )     
        for tree in rf.estimators_:
            Features = np.array(tree.tree_.feature,dtype = 'int32' )
            paths = tree.decision_path( np.array(X[tree.oob_ind]) ).toarray()  
            for case,path in zip(tree.oob_ind,paths):
                features = Features[np.where(path)[0]]
                feaCounter = Counter(features)
                feaCounter.pop(-2)
                if tree.predict(X[case,:].reshape(1,-1)) == float(y[case]):
                    for k,w in feaCounter.items():
                        sample[case,k] += w
                else:
                    for k,w in feaCounter.items():
                        sample[case,k] -= w
        mea = np.mean(sample,axis=0)/rf.n_estimators
#        print(sample,mea)
        return mea
    





class rf_select(importanceEstimator):
    '''
        Here be documentation
    '''
    def __init__(self,clf=None,recursive=True,importance='mcm_'):
        # Initialise
        self.clf_=clf
        self.recursive_= recursive
        
        if importance in ['gini','permutation','mcm','mcm_']:
            self.importance_= importance
        else:
            raise ValueError('Error: importance not recognised: '+str(importance))
        
        #
        # Run
    def select(self,recursive,X,y):
        if self.recursive_:
            return self._recursiveFeatureElimination(X,y)
        else:
            return self._staticFeatureElimination(X,y)


    def _recursiveFeatureElimination(self,X,y):
        #
        # Elminate features using updated importance
        resultDict= {}
#        toCut= [] #start with all features
#        ctr= 0
        oob_s = []
        #
        # Keeps minimum of 3 most important features
#        while ctr < np.shape(X)[1] - 3:
#            toUse= filter(lambda x: True if x not in toCut else False,range(np.shape(X)[1]))
#            #
#            # Re-evaluate importances and metric using allowed features
##            self.clf_.fit(X[:,toUse] , y)
#            Imp_i= importanceEstimator(self.clf_,X[:,toUse] , y,algorithm=self.importance_)
#            importances_i= Imp_i.fit( X[:,toUse] , y )
#            resultDict[ctr]= ( toUse, importances_i )
#            oob_s.append(self.clf_.oob_score_)
#            #
#            # cut lowest importance feature next time
#            toCut.append( np.argsort( importances_i )[0] ) 
#            ctr+=1
        
        
        Imp= importanceEstimator(self.clf_,X,y,algorithm=self.importance_)
        importances= Imp.fit( X,y )
        ordering = np.argsort(importances)
        ordering = list(ordering)
        toUse = ordering[0:len(ordering)]
        self.clf_.fit(X[:,toUse],y)
        oob_s.append(self.clf_.oob_score_)
        resultDict[0] = (toUse,importances)
        
        
        for ind in xrange(np.shape(X)[1]/10-1):
#            ctr = ind*10
            toUse = ordering[10:len(ordering)]
            Imp_i= importanceEstimator(self.clf_,X[:,toUse],y,algorithm=self.importance_)
            importances_i= Imp_i.fit( X[:,toUse],y )
            ordering = np.argsort(importances_i)
            ordering = list(ordering)
            self.clf_.fit(X[:,toUse],y)
            oob_s.append(self.clf_.oob_score_)
            resultDict[ind+1] = (toUse,importances_i)
        
        return resultDict,oob_s

    def _staticFeatureElimination(self,X,y):
        #
        # Get initial importances
        Imp= importanceEstimator(self.clf_,X, y,algorithm=self.importance_)
        importances= Imp.fit(X,y)
        
        # Elminate features using static importance
        ordering= np.argsort( importances )
        ordering = list(ordering)
        resultDict= {}
#        toCut= []
        oob_s = []
#        for ind,i in enumerate(ordering[:-2]):
#            toUse= filter(lambda x: True if x not in toCut else False,range(len(ordering)))
#            #
##            Imp_i = copy.deepcopy( Imp )
##            = Imp_i.fit(X[:,toUse],y)
#            self.clf_.fit(X[:,toUse],y)
#            oob_s.append(self.clf_.oob_score_)
#            resultDict[ind]= ( toUse,self.clf_.oob_score_ )
#            #
#            toCut.append( i )
        
        for ind in xrange(np.shape(X)[1]/10):
            ctr = ind*10
            toUse = ordering[ctr:len(ordering)]
#            toUse = list(toUse)
#            Imp_i = copy.deepcopy( Imp )
#            importances_i = Imp_i.fit(X[:,toUse],y)
            
            self.clf_.fit(X[:,toUse],y)
            oob_s.append(self.clf_.oob_score_)
            resultDict[ind] = (toUse,self.clf_.oob_score_)
        return resultDict,oob_s    
    
    
if __name__ == "__main__":
#    iris = load_breast_cancer()
#    idx = range(len(iris.data))
#    np.random.shuffle(idx)
#    X = iris.data[idx]
#    Y= iris.target[idx]
    
#    输入.mat格式文件
#    import scipy.io as sio
#    data = sio.loadmat('centralNervousSystemoutcome.mat')
#    X = data.get('data')
#    Y = data.get('datalabel')
#    Y = Y.reshape(np.shape(Y)[0],)
    
    
    
    rootdir = os.getcwd()
    data1 = []
    with open(rootdir+'\\colon.txt') as file:
        for line in file:
            data1.append(line.split(' '))
        
        npdata = np.array(data1[0:])
        npdata = npdata
    X,Y = split_data(npdata)
   
   
#    Y = np.array(Y)
#    
    n_estimators =500
    rf = forest.RandomForestClassifier(n_estimators=n_estimators,oob_score=True,n_jobs=-1)
    
    
#    impEst = importanceEstimator(rf,X,Y,'mcm_')
#    print impEst.fit(X,Y)
    
    
    oob_score1 = []
    oob_score2 = []
    result1 = []
    result2 = []
    for t in xrange(10):
        sel_1 = rf_select(rf,True,'gini')
        result_1,oob_s1 = sel_1.select(True,X,Y)
        oob_score1.append(oob_s1)
        result1.append(result_1)
        
        sel_2 = rf_select(rf,False,'gini')
        result_2,oob_s2 = sel_2.select(False,X,Y)
        oob_score2.append(oob_s2)
        result2.append(result_2)
        
    oob_score1 = np.array(oob_score1)
    oob_score2 = np.array(oob_score2)
    mean1 = np.mean(oob_score1,axis = 0)
    mean2 = np.mean(oob_score2,axis = 0)
    
    max1 = np.max(oob_score1,axis=1)
    max2 = np.max(oob_score2,axis=1)
    
    ave1 = np.mean(max1)
    ave2 = np.mean(max2)
    
    print ave1,ave2
    
    
    
#    plt.plot(range(1,len(max)+1),)
#    plt.rcParams['font.sans-serif'] = ['SimHei']
#    plt.rcParams['axes.unicode_minus'] = False
#    plt.xlabel(u'变量个数')
#    plt.ylabel('oob_score')
#    plt.show()
    
    
    
    
    
    
    
    
    
    