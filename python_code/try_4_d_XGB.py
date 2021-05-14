# -*- coding: utf-8 -*-
"""
Created on Tue Apr 13 23:35:29 2021

@author: aishe
"""
import xgboost as xgb
import pandas as pd
import numpy as np
import traceback
import sys

from collections import Counter
from sklearn.preprocessing import MinMaxScaler
from sklearn.model_selection import train_test_split
from sklearn.metrics import confusion_matrix
import matplotlib.pyplot as plt
import pickle
params = {}
params['feature_path']='E:/2021spr_term_class/课程——移动互联网应用设计/互移课设数据/paderborn-train/smote_feature/trainfea_resampled_N15_M01_F10.csv'

params['test_path']='E:/2021spr_term_class/课程——移动互联网应用设计/互移课设数据/paderborn-train/smote_feature/trainfea_resampled_N15_M07_F10.csv'

#测验模型的一些超参数
par2 = {
    'booster': 'gbtree',    #迭代模型选择
    'silent': 0,    #设置为0，可输出信息
    'nthread': 4,    #线程数
    
    'objective': 'multi:softmax',    #需要被最小化的损失函数，选的是多分类预测类别
    'num_class': 5,    #指定类别数目
    
    'gamma': 0.1,    #指定节点分裂所需的最小损失函数下降值
    'subsample': 0.65,    #控制对于每棵树的随机采样比例，一般为0.5-1
    'colsample_bytree': 0.7,    #控制每棵树随机采样的列数占比，每一列是一个特征
    'alpha':0.05,
    'lambda': 0.05,    #权重的L2正则项
    'max_depth': 4,    #树的最大深度，一般为3-10
    'min_child_weight': 1,    #最小叶子节点样本权重和
    'eta': 0.1,    #学习率，默认0.3，一般为0.01-0.2
    
    'seed': 1000,    #随机种子
    'eval_metric': 'merror',    #多分类的错误率
}
plst = list(par2.items())
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

    #归一化到0，1之间
    scaled_fea = MinMaxScaler().fit_transform(ar_fea)
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
    
    d = feature_distance(scaled_fea, labels)
    d_th =0.4
    N_d = np.sum(d>=d_th)
    d_col = col[np.argsort(-d)]  #按照距离降序排列的特征名
    N_d_col = d_col[:N_d]        #待测验的前N_d个特征名
    
    data_1 = pd.read_csv(params['test_path'])
    fea_1 = np.array(data_1)
    ar_fea_1 = fea_1[:,:-1]
    col_1 = np.array(data_1.columns)[:-1]
    labels_1 = fea_1[:,-1]
    
    cfu_matrics = np.zeros((N_d,5,5))  #标签总数为5
    for i in range(1,N_d+1):
        
        
        X_train = ar_fea[:,np.argsort(-d)[:i]]
        y_train = labels
        
        X_test = ar_fea_1[:,np.argsort(-d)[:i]]
        y_test = labels_1
        
        #X_train, X_test, y_train, y_test = train_test_split(
                #fea_d, y, test_size=0.3, random_state=42)
        ############
        dtrain = xgb.DMatrix(X_train, y_train)
        num_rounds = 112
        model = xgb.train(plst, dtrain, num_rounds)
        #保存训练好的模型
        string_i = 'pima.pickle_' + str(i) + '.dat'
        pickle.dump(model,open(string_i,'wb'))
        # 对测试集进行预测
        dtest = xgb.DMatrix(X_test)
        predict = model.predict(dtest)
        
        cfu_matrics[i-1] = confusion_matrix(y_test,predict)
    #评分标准，混淆矩阵，准确率以及加权准确率
    def w_score(cfu_matric):
        
        acu = np.array([cfu_matric[i,i]/np.sum(cfu_matric[i]) for i in range(len(cfu_matric))])
        w = np.array([0.1,0.2,0.25,0.2,0.25])
        v = np.sum(acu*w)
        print('混淆矩阵：',cfu_matric,sep='\n')
        print('每个类别的预测准确率：',acu,sep='\n')
        print('加权后的score：',v,sep='\n')
        
        return v
    v = []
    for j in range(N_d):
        print(j)
        v.append(w_score(cfu_matrics[j]))
    k = [i for i in range(1,N_d+1)]
    plt.scatter(k,v)
    plt.show()
    
    new_fea = ar_fea[:,d>=0.4]
    new_col = list(col[d>=0.4]) + ['label']
    #
    labels = labels.reshape((-1,1))
    data_new = np.hstack((new_fea,labels))
    result = pd.DataFrame(data_new, columns = new_col)
    #result.to_csv(params['save_path'], sep=',', header=True, index=False)
    
    
except Exception as e:
    traceback.print_exc()
    print(e)