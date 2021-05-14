# -*- coding: utf-8 -*-
"""
Created on Thu Apr  1 19:23:46 2021

@author: aishe
"""
import pandas as pd
import traceback
import sys
#d>=0.6
fea_6 = ['time_mean',
         'time_median',
         'freq_median',
         'freq_f5',
         'freq_f8',
         'ratio_cD1',]
new_fea_6=['time_median',
           'time_mean',
           'freq_f5',
           'freq_f8',
           'ratio_cD1',
           'freq_median']

params = {}
params['path'] = 'E:/2021spr_term_class/课程——移动互联网应用设计/互移课设数据/paderborn-test2/small_testfea_2_N15_M07_F04.csv'
params['opath'] = 'E:/2021spr_term_class/课程——移动互联网应用设计/互移课设数据/paderborn-test2/small_testfea_6_2_N15_M07_F04.csv'
#params['path'] = 'E:/2021spr_term_class/课程——移动互联网应用设计/互移课设数据/paderborn-train/smote_feature/trainfea_resampled_N15_M01_F10.csv'
#params['opath'] = 'E:/2021spr_term_class/课程——移动互联网应用设计/互移课设数据/1.csv'

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

    dfs = pd.read_csv(params['path'])
    dfg = dfs.loc[:,fea_6]
    new_df = dfg.reindex(new_fea_6, axis='columns')

    new_df.to_csv(params['opath'], sep=',', header=True, index=False)
  
except Exception as e:
    traceback.print_exc()
    print(e)