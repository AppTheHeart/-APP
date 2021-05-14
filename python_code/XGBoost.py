# -*- coding: utf-8 -*-
"""
Created on Mon Apr  5 12:47:41 2021

@author: aishe
"""


import xgboost as xgb
from xgboost import plot_importance
from matplotlib import pyplot as plt
import pandas as pd
import numpy as np
from sklearn.metrics import confusion_matrix,precision_score, recall_score, accuracy_score, f1_score


import sys
import json
import traceback
import joblib



class Result:
    precision = 0
    recall = 0
    accuracy = 0
    rocX = []
    rocY = []
    featureImportances = []

params = {}

params['train_path'] = 'E:/2021spr_term_class/课程——移动互联网应用设计/互移课设数据/paderborn-train/smote_feature/trainfea_4_N15_M01_F10.csv'
params['test_path'] = 'E:/2021spr_term_class/课程——移动互联网应用设计/互移课设数据/paderborn-train/smote_feature/trainfea_4_N15_M07_F10.csv'
par = {
    'booster': 'gbtree',    #迭代模型选择
    'silent': 0,    #设置为0，可输出信息
    'nthread': 4,    #线程数
    
    'objective': 'multi:softmax',    #需要被最小化的损失函数，选的是多分类预测类别
    'num_class': 5,    #指定类别数目
    
    'gamma': 0.1,    #指定节点分裂所需的最小损失函数下降值
    'subsample': 0.7,    #控制对于每棵树的随机采样比例，一般为0.5-1
    'colsample_bytree': 0.7,    #控制每棵树随机采样的列数占比，每一列是一个特征
    'lambda': 2,    #权重的L2正则项
    'max_depth': 6,    #树的最大深度，一般为3-10
    'min_child_weight': 3,    #最小叶子节点样本权重和
    'eta': 0.1,    #学习率，默认0.3，一般为0.01-0.2
    
    'seed': 1000,    #随机种子
    'eval_metric': 'merror',    #多分类的错误率
}

num_rounds = 112
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

    train = np.array(pd.read_csv(params['train_path']))
    train_x = train[:,:-1]
    train_y = train[:,-1]
    
    test = np.array(pd.read_csv(params['test_path']))
    test_x = test[:,:-1]
    test_y = test[:,-1]
    
    plst = list(par2.items())

    dtrain = xgb.DMatrix(train_x, train_y)
    num_rounds = 112
    model = xgb.train(plst, dtrain, num_rounds)
    #保存训练好的模型
    joblib.dump(model,'pima.pickle.dat')
    
    # 对测试集进行预测
    dtest = xgb.DMatrix(test_x)
    predict = model.predict(dtest)


    precision = precision_score(test_y, predict,average = 'micro')
    recall = recall_score(test_y, predict,average = 'micro')
    accuracy = accuracy_score(test_y, predict)
    res = {}
    res['precision'] = precision
    res['recall'] = recall
    res['accuracy'] = accuracy
    res['fMeasure'] = f1_score(test_y, predict,average = 'micro')
    
    #res['rocArea'] = roc_auc_score(test_y, predict,multi_class,average = 'macro')
    #res['featureImportances'] = clf.feature_importances_.tolist()
    print(json.dumps(res))
    def w_score(test_y,predict):
        cfu_matr = confusion_matrix(test_y,predict)
        acu = np.array([cfu_matr[i,i]/len(test_y[test_y==i]) for i in range(len(cfu_matr))])
        w = np.array([0.1,0.2,0.25,0.2,0.25])
        print('混淆矩阵：',cfu_matr,sep='\n')
        print('每个类别的预测准确率：',acu,sep='\n')
        print('加权后的score：',np.sum(acu*w),sep='\n')
    w_score(test_y,predict)
    plot_importance(model)
    plt.show()
except Exception as e:
    traceback.print_exc()