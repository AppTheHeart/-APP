# -*- coding: utf-8 -*-
"""
Created on Mon Apr  5 13:40:56 2021

@author: aishe
"""
from sklearn.metrics import precision_score, recall_score, accuracy_score, f1_score
import xgboost as xgb
import pandas as pd
import numpy as np
import warnings
warnings.filterwarnings('ignore')
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
params['model'] ='E:/2021spr_term_class/课程——移动互联网应用设计/大作业/XGBoost_model.model'

params['test'] = 'E:/2021spr_term_class/课程——移动互联网应用设计/互移课设数据/1.csv'
params['opath'] = 'E:/2021spr_term_class/课程——移动互联网应用设计/互移课设数据/2.csv'


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
    

    model = joblib.load(params['model'])
    
    test_csv = pd.read_csv(params['test'])
    test_feature = test_csv.drop(['label'], axis=1)
    tdata = np.array(test_feature)
    
    test_y = test_csv['label']
    #test_y = test[:,-1]
    
    
    


    # 对测试集进行预测
    dtest = xgb.DMatrix(tdata)
    #y_pred = model.predict_proba(dtest)
    y_pred = model.predict(dtest)
#    Predict = [i for i in range(tdata.shape[0])]
#    for i in range(tdata.shape[0]):
#        # Predict = predict[i]
#        Predict[i] = int(y_pred[i])
        
    
    precision = precision_score(test_y, y_pred,average='macro')
    recall = recall_score(test_y, y_pred,average='macro')
    accuracy = accuracy_score(test_y, y_pred)
    
    res = {}
    res['precision'] = precision
    res['recall'] = recall
    res['accuracy'] = accuracy
    res['fMeasure'] = f1_score(test_y, y_pred,average='macro')
        #res['rocArea'] = roc_auc_score(test_y.reshape((-1,1)), y_pred.reshape((-1,1)),average='macro',multi_class='ovo')
    res['rocArea'] =0
    #res['featureImportances'] = list(model.get_fscore().items())
    print(json.dumps(res))

    predict_df = pd.DataFrame(y_pred,columns = ['predict'])
    predict_df.to_csv(params['opath'], sep=',', header=True, index=False)
    
except Exception as e:
    traceback.print_exc()
    print(e)
