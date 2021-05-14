# -*- coding: utf-8 -*-
"""
Created on Wed Mar 24 20:11:19 2021

@author: aishe
"""
from collections import Counter
from imblearn.over_sampling import SMOTE
import pandas as pd
import numpy as np
import sys
import csv


params = {}
params['ratio'] = 0.25
params['kneighbors'] = 5
params['path'] = '/usr/local/data/data.csv'
params['opath'] = '/usr/local/data/data_out.csv'
argvs = sys.argv

try：
    #用sys.argv读取命令行中传递过来的参数
    for i in range(len(argvs)):
        if i < 1:
            continue
        if argvs[i].split('=')[1] == 'None':
            params[argvs[i].split('=')[0]] = None
        else:
            Type = type(params[argvs[i].split('=')[0]])
            params[argvs[i].split('=')[0]] = Type(argvs[i].split('=')[1])
    
    #读取数据，存属性
    dfs = pd.read_csv(params['data_path'])
    col = list(dfs.columns)
    data = np.array(dfs.iloc[:,:-1])
    labels = np.array(dfs['label'])
    
    #统计原数据的标签数量
    c1 = Counter(labels)
    #根据指定的ratio（少数样本/多数样本），来生成ra集合，用以在SMOTE的指定创建
    m = max(c1.values())
    ra = {}
    for i in c1.keys():
        if (c1[i]/m) < params['ratio']:
            ra[i] = int(m*params['ratio'])

    #对样本进行非均衡处理
    smo = SMOTE(sampling_strategy=ra,
                k_neighbors=params['kneighbors'],
                random_state=42)
    data_smote,labels_smote = smo.fit_resample(data,labels)
    #c2_smote = Counter(labels_smote)

    #存储过采样后的数据
    labels_smote = np.array(labels_smote).reshape(-1,1)
    data_label = np.hstack((data_smote,labels_smote))
    
    result = pd.DataFrame(data_label, columns = col)
    result.to_csv(params['save_path'], sep=',', header=True, index=False)
except Exception as e:
    print(e)