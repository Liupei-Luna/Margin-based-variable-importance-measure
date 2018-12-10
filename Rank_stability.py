# -*- coding: utf-8 -*-
"""
Spyder Editor

This is a temporary script file.
"""

# -*- coding: utf-8 -*-
"""
Created on Wed May 16 16:29:42 2018

@author: Dell
"""


import numpy as np
import sys
import copy
import os

from sklearn.cross_validation import ShuffleSplit,KFold
from sklearn.metrics import r2_score
from collections import defaultdict
from scipy.stats import pearsonr,spearmanr
from sklearn.metrics import jaccard_similarity_score
from scipy.spatial.distance import cosine


from sklearn.ensemble import forest
from sklearn.tree import DecisionTreeRegressor, DecisionTreeClassifier, _tree
reload(forest)
from sklearn.datasets import load_breast_cancer, load_iris
from collections import Counter
from matplotlib import pyplot as plt

from scipy.spatial.distance import pdist
import pandas as pd


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

def margin_oob(rf,x,y):
            n_samples= x.shape[0]
            n_classes= len( set(y) )
            A = np.zeros((n_samples,n_classes))
            for tree in rf.estimators_:
                J = tree.predict(x[tree.oob_ind,:])
                J = np.array(J,dtype='int32')
                for count,instance in enumerate (tree.oob_ind):
                    A[instance][J[count] ] = 1 + A[instance][J[count] ] 
        #print(np.sum(np.sum(B))/150/10)
                    
    
        #oob样本margin
            margin_oob=[]
            
            def _marginFun(X):
                L,y = X
                y = y-1  #python从0开始
        #        print (L,y)
                s = np.sum(L) #分母
                if s == 0:
        #        print('############################################')
                    return np.NAN
                a = L[y] #分子被减数，即实际值对应票数
                L[y] = 0  #将实际值置为0，在选择最大值时不会被选中
                return float(a- np.max(L))/s
            
            margin_oob =map( _marginFun, zip (A.copy(),y) )
        
            return margin_oob
        
def margin_all(rf,x,y):
        
        n_samples,n_classes= x.shape

    
        B=np.zeros((n_samples,n_classes))        
        for tree in rf.estimators_:
            J = tree.predict(x)
            J = np.array(J,dtype='int32')
            for i in range(x.shape[0]):
                B[i,J[i]] = 1 + B[i,J[i]]
    #print(np.sum(np.sum(B))/150/10)
                
    #所有样本margin
        margin_all=[]
        def _marginFun(X):
            L,y = X
            y = y-1  #python从0开始
    #        print (L,y)
            s = np.sum(L) #分母
            if s == 0:
                return np.NAN
            a = L[y] #分子被减数，即实际值对应票数
            L[y] = 0  #将实际值置为0，在选择最大值时不会被选中
            return float(a- np.max(L))/s
        margin_all =map( _marginFun, zip (B.copy(),y) )
        
        
        return margin_all 

class importanceEstimator():
    '''
        Here be documentation.
    '''
    def __init__(self,clf,algorithm='gini'):
        
        
#        clf.fit(X_train,Y_train)
        
        self.clf= clf
#        self.clf.fit(X_train,Y_train)
        if algorithm in ['gini','permutation','mcm','mcm_','margin_all']:
            self.algorithm= algorithm
        else:
            raise ValueError('Error: algorithm not recognised: '+str(algorithm))


    def fit(self,X_test,Y_test):
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
#        n_folds = 10
#        kf=KFold(len(X),n_folds=n_folds,random_state=1)
#        scores = defaultdict(list) # Any unknown element is automatically a list
        rf= copy.deepcopy(self.clf)
        #
        #crossvalidate the scores on a number of different random splits of the data
#        importances = defaultdict(list) 
#        for train_idx, test_idx in kf:
#            X_train, X_test = X[train_idx], X[test_idx]
#            y_train, y_test = y[train_idx], y[test_idx]
#            rf.fit(X_train, y_train)
#            
#            if self.algorithm == 'gini':
#                importances[i].append( self.giniImportance(rf,X_test,y_test) )
#            elif self.algorithm == 'permutation':
#                importances[i].append( self.permutationImportance(rf,X_test,y_test) )
#            elif self.algorithm == 'mcm':
#                importances[i].append( self.mcmImportance(rf,X_test,y_test) )
#            elif self.algorithm == 'mcm_':
#                importances[i].append( self.mcmImportance_(rf,X_test,y_test) )
            
            
#            
        if self.algorithm == 'gini':
            importances = self.giniImportance(rf,X_test,Y_test) 
        elif self.algorithm == 'permutation':
            importances = self.permutationImportance(rf,X_test,Y_test) 
        elif self.algorithm == 'mcm':
            importances = self.mcmImportance(rf,X_test,Y_test)
        elif self.algorithm == 'mcm_':
            importances = self.mcmImportance_(rf,X_test,Y_test)
        elif self.algorithm == 'margin_all':
            importances = self.margin_allImportance(rf,X_test,Y_test)
        
        
        # Return mean importance and metric
#        importances= np.array([np.mean(scores[i]) for i in range(X.shape[1])])
        return importances
            
            

    def giniImportance(self,rf,X,y):
        return rf.feature_importances_

    def permutationImportance(self,rf,X,y):
        # Get feature importances
        acc = r2_score(y,rf.predict(X))
        scores= []
        for i in range(X.shape[1]):
            X_t = X.copy()
            np.random.shuffle(X_t[:, i])
            shuff_acc = r2_score(y,rf.predict(X_t))
            scores.append((acc-shuff_acc)/acc)
        return np.array(scores)
    
    
    
    
    def mcmImportance(self,rf,X,y):
      
        sample = np.zeros ( X.shape )     
        for tree in rf.estimators_:
            Features = tree.tree_.feature
            for i,case in enumerate(X):
                feaCounter = Counter(Features)
                feaCounter.pop(-2)
                if tree.predict(case.reshape(1,-1)) == float(y[i]):
                    for k,w in feaCounter.items():
                        sample[case,k] += w
                else:
                    for k,w in feaCounter.items():
                        sample[case,k] -= w
        mea = np.mean(sample,axis=0)/rf.n_estimators
        return mea
    
    def mcmImportance_(self,rf,X,y):
        inds = []
        instances = []
        for ind,instance in enumerate(X):
            inds.append(ind)
            instances.append(instance)
      
        sample = np.zeros ( X.shape )     
        for tree in rf.estimators_:
            Features = np.array(tree.tree_.feature,dtype = 'int32' )
            paths = tree.decision_path( np.array(X)).toarray()  
            for i,(case,path) in enumerate(zip(X,paths)):
                
                features = Features[np.where(path)[0]]
                feaCounter = Counter(features)
                feaCounter.pop(-2)
                
                if tree.predict(case.reshape(1,-1)) == float(y[i]):
                    for k,w in feaCounter.items():
                        sample[i,k] += w
                else:
                    for k,w in feaCounter.items():
                        sample[i,k] -= w
        mea = np.mean(sample,axis=0)/rf.n_estimators
#        print(sample,mea)
        return mea
    
    
#    def margin_oobImportance(self,rf,X,y):
#        
#        
#        scores= defaultdict(list)
#        for i in range(X.shape[1]):
#            X_t = X.copy()
#            np.random.shuffle(X_t[:, i])
#            sim_i = cosine(margin_all(rf,X,y),margin_all(rf,X_t,y))
##            sim_2 = pearsonr(margin_oob_before,margin_oob_after)
##            sim_3,pvalue = spearmanr(margin_oob_before,margin_oob_after,axis=0)
#            scores[i].append(sim_i)
#        return np.array([ np.mean(scores[i]) for i in range(X.shape[1]) ])
        
    
    
    def margin_allImportance(self,rf,x,y):
        
        
        scores= defaultdict(list)
        for i in range(x.shape[1]):
            X_t = x.copy()
            np.random.shuffle(X_t[:, i])
            sim_i = cosine(margin_all(rf,x,y),margin_all(rf,X_t,y))
#            sim_2 = pearsonr(margin_oob_before,margin_oob_after)
#            sim_3,pvalue = spearmanr(margin_oob_before,margin_oob_after,axis=0)
            scores[i].append(sim_i)
        return np.array([ np.mean(scores[i]) for i in range(X.shape[1]) ])





class rf_select(importanceEstimator):
    '''
        Here be documentation
    '''
    def __init__(self,clf=None,recursive=True,importance='mcm_'):
        # Initialise
        self.clf_=clf
        self.recursive_= recursive
        
        if importance in ['gini','permutation','mcm','mcm_','margin_all']:
            self.importance_= importance
        else:
            raise ValueError('Error: importance not recognised: '+str(importance))
        
        #
        # Run
    def select(self,X_train,y_train,X_test,y_test):
        self.clf_.fit(X_train,y_train)
        if self.recursive_:
            return self._recursiveFeatureElimination(X_train,y_train,X_test,y_test)
        else:
            return self._staticFeatureElimination(X_train,y_train,X_test,y_test)


    def _recursiveFeatureElimination(self,X_train,y_train,X_test,y_test):
        #
        # Elminate features using updated importance
        resultDict = {}
        toCut = []
        ctr= 0
        scores = []    
        while ctr < np.shape(X)[1]-3:
            toUse= filter(lambda x: True if x not in toCut else False,
                          range(np.shape(X)[1]))
            self.clf_.fit(X_train[:,toUse],y_train)
            
            Imp_i= importanceEstimator(clf=self.clf_,algorithm=self.importance_)
            importance_i = Imp_i.fit(X_test,y_test)
            resultDict[ctr]= ( toUse,importance_i)
            toCut.append(np.argsort(importance_i)[0])
            
            scores.append(r2_score(y_test,self.clf_.predict(X_test[:,toUse])))
            ctr +=1
            
        
        return resultDict,scores

    def _staticFeatureElimination(self,X_train,y_train,X_test,y_test):
        #
        # Get initial importances
        
        Imp= importanceEstimator(self.clf_,algorithm=self.importance_)
        importances= Imp.fit(X_test,y_test)
        #
        # Elminate features using static importance
        ordering= np.argsort( importances )
        resultDict= {}
        toCut= []
        scores = []
     
        for ind,i in enumerate(ordering[:-2]):

            toUse = filter(lambda x: True if x not in toCut else False,range(len(ordering)))
 
            self.clf_.fit(X_train[:,toUse],y_train)
            scores.append(self.clf_(X_test[:toUse],y_test))
            resultDict[ind] = (toUse,self.clf_(X_test[:toUse],y_test))
            toCut.append(i)
            
          
        return resultDict,scores    

#Kuncheva coefficient
 
   
if __name__ == "__main__":

#    iris = load_breast_cancer()
#    idx = range(len(iris.data))
#    np.random.shuffle(idx)
#    X = iris.data[idx]
#    Y= iris.target[idx]
    
    
    
    
    
#    输入.mat格式文件
#    import scipy.io as sio
#    data = sio.loadmat('Lung_Cancer.mat')
#    X = data.get('data')
#    Y = data.get('datalabel')
#    Y = Y.reshape(np.shape(Y)[0],)
    
    
    
    rootdir = os.getcwd()
    data1 = []
    with open(rootdir+'\\p_gene.txt') as file:
        for line in file:
            data1.append(line.split('\t'))
        
        npdata = np.array(data1[0:])
        npdata = npdata
    X,Y = split_data(npdata)
    Y = np.array(Y)
    Y = np.int32(Y)
#    
    n_estimators =100
    rf = forest.RandomForestClassifier(n_estimators=n_estimators,n_jobs=-1)
    
    

    n_folds = 10
    kf=KFold(len(X),n_folds=n_folds,random_state=1)
    
    
    sel_1 = rf_select(rf,True,'gini')
#    sel_2 = rf_select(rf,False,'mcm_')
#    sel_3 = rf_select(rf,False,'permutation')
#    sel_4 = rf_select(rf,False,'margin_all')
    
    
    rank1 = []
    score1 = []
#    rank2 = []
#    score2 = []
#    rank3 = []
#    score3 = []
#    rank4 = []
#    score4 = []
    for train_idx, test_idx in kf:
            X_train, X_test = X[train_idx], X[test_idx]
            y_train, y_test = Y[train_idx], Y[test_idx]
            rank_1,score_1 = sel_1.select(X_train,y_train,X_test,y_test)
            rank1.append(rank_1)
            score1.append(score_1)
            rank_2,score_2 = sel_2.select(X_train,y_train,X_test,y_test)
            rank2.append(rank_2)
            score2.append(score_2)
            
            rank_3,score_3 = sel_3.select(X_train,y_train,X_test,y_test)
            rank3.append(rank_3)
            score3.append(score_3)
            rank_4,score_4 = sel_4.select(X_train,y_train,X_test,y_test)
            rank4.append(rank_4)
            score4.append(score_4)

    score1_sum = np.mean(score1,axis=0)
    score2_sum = np.mean(score2,axis=0)
    score3_sum = np.mean(score3,axis=0)
    score4_sum = np.mean(score4,axis=0)
    

    
    similarity1_k = []
    similarity2_k = [] 
    similarity3_k = []
    similarity4_k = []
    for i in xrange(n_folds-1):
        for j in xrange(1,n_folds):
#            
            
            cor1_k = jaccard_similarity_score(rank1[i][39][0],rank1[j][39][0])
            cor2_k = jaccard_similarity_score(rank2[i][39][0],rank2[j][39][0])
            cor3_k = jaccard_similarity_score(rank3[i][39][0],rank3[j][39][0])
            cor4_k = jaccard_similarity_score(rank4[i][39][0],rank4[j][39][0])
            
            similarity1_k.append(cor1_k)
            similarity2_k.append(cor2_k)
            similarity3_k.append(cor3_k)
            similarity4_k.append(cor4_k)
        

    
    stability1_k = 2*np.sum(similarity1_k)/(n_folds*(n_folds-1))
    stability2_k = 2*np.sum(similarity2_k)/(n_folds*(n_folds-1))
    stability3_k = 2*np.sum(similarity3_k)/(n_folds*(n_folds-1))
    stability4_k = 2*np.sum(similarity4_k)/(n_folds*(n_folds-1))
    
    #十折交叉每次特征选择结果准确率均值()
    
    score1 = np.array(score1)
    score2 = np.array(score2)
    score3 = np.array(score3)
    score4 = np.array(score4)
    
    score1_ave = np.mean(score1,axis=0)
    score2_ave = np.mean(score2,axis=0)
    score3_ave = np.mean(score3,axis=0)
    score4_ave = np.mean(score4,axis=0)
    
    x_1 = np.argsort(score1_ave)
    x_2 = np.argsort(score2_ave)
    print score1_ave[x_1[-1]],score2_ave[x_1[-1]],score3_ave[x_1[-1]],score4_ave[x_1[-1]]
    print score1_ave[x_2[-1]],score2_ave[x_2[-1]],score3_ave[x_2[-1]],score4_ave[x_2[-1]]
#    
#    #13行4列，行代表13种选择特征个数，列表示四种重要性度量方法得到的结果，数据是十折交叉准确率结果
#    tack_z = [[],[],[],[]]
#    tack_z[0]= score1_ave
#    tack_z[1]= score2_ave
#    tack_z[2]= score3_ave
#    tack_z[3]= score4_ave
#    tack_z = np.array(tack_z)
#    tack_z = tack_z.T
#            
#    
#    
#    
#
#
#    
#    
#    k = range(1,len(score2_sum)+1)
#    plt.plot(k,score4_sum,'ko')
#    plt.plot(k[:],score4_sum[:])
#    plt.rcParams['font.sans-serif'] = ['SimHei']
#    plt.rcParams['axes.unicode_minus'] = False
#    plt.xlabel(u'特征选择迭代次数')
#    plt.ylabel('accuracy')
#    plt.show()
#    
            

    
    
    
    
    
    
    