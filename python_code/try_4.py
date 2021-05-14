# -*- coding: utf-8 -*-
"""
Created on Wed Mar 31 22:22:28 2021

@author: aishe
"""


import pandas as pd
import numpy as np
import traceback
import sys

from collections import Counter
from sklearn.preprocessing import MinMaxScaler
params = {}
params['feature_path']='E:/2021spr_term_class/课程——移动互联网应用设计/互移课设数据/paderborn-train/smote_feature/trainfea_resampled_N15_M01_F10.csv'
params['save_path'] = 'E:/2021spr_term_class/课程——移动互联网应用设计/互移课设数据/paderborn-train/smote_feature/trainfea_4_N15_M01_F10.csv'

argvs = sys.argv
try:
    for i in range(len(argvs)):
        if i < 1:
            continue
        if argvs[i].split('=')[1] == 'None':
            params[argvs[i].split('=')[0]] = None
        else:
            Type = type(params[argvs[i].split('=')[0]])
            params[argvs[i].split('=')[0]] = Type(argvs[i].split('=')[1])
    #读取特征+标签+特征名
    data = pd.read_csv(params['feature_path'])
    fea = np.array(data)
    ar_fea = fea[:,:-1]
    col = np.array(data.columns)[:-1]
    labels = fea[:,-1]

    
    #单个特征的分离距离计算
    def signal_feature_distance(X, y, j, m_j):
        assert 1<= j <= X.shape[1], 'the j must be from one to the number of X\'s feature.'
        assert X.shape[0] == y.shape[0], 'the number of X must be equal to the number of y.'

        #统计数据的标签的种类及相应的数量
        c1 = Counter(y)
        Nc = len(c1)  #标签种类的数量
        if len(m_j) != Nc:
            return 0
        
        d_j = 0
        for k in range(Nc):
            n_k = c1[k]
            X_k = X[y==k]
            X_jk = X_k[:,j-1]
            d_jk = 0
            
            for c in range(Nc):
                d_jk += np.sum( (X_jk - m_j[c])*(X_jk - m_j[c]) )
                #print(j,'success of c\'s m_jc',c)
                
            d_j += d_jk / (Nc*n_k)
            print(j,'success of all k\'s end',k)
        
        return d_j/Nc
    
    #对X的所有特征进行分离距离的计算
    def feature_distance(X, y):
        assert X.shape[0] == y.shape[0], 'the number of X must be equal to the number of y.'
        
        Q = X.shape[1]
        d = []
        
        c1 = Counter(y)
        #第c类的第j个特征的均值
        m_jc = np.zeros((len(c1), Q))
        for i in range(len(c1)):
            m_jc[i,:] = np.sum(X[y==i],axis=0)/c1[i] 
        
        for j in range(1,Q+1):
            d.append(signal_feature_distance(X, y, j, m_jc[:,j-1]))
        
        d = np.array(d)
        return d/np.max(d)
    #归一化到0，1之间
    scaled_fea = MinMaxScaler().fit_transform(ar_fea)
    d = feature_distance(scaled_fea, labels)
    new_fea = ar_fea[:,d>=0.4]
    new_col = list(col[d>=0.4]) + ['label']
    #
    labels = labels.reshape((-1,1))
    data_new = np.hstack((new_fea,labels))
    result = pd.DataFrame(data_new, columns = new_col)
    result.to_csv(params['save_path'], sep=',', header=True, index=False)
    
    
except Exception as e:
    traceback.print_exc()
    print(e)