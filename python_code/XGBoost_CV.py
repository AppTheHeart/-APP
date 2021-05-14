# -*- coding: utf-8 -*-
"""
Created on Sun Apr 11 16:37:01 2021

@author: aishe
"""

import xgboost as xgb
from xgboost.sklearn import XGBClassifier
import pandas as pd
import numpy as np
from sklearn.model_selection import GridSearchCV
from sklearn.metrics import confusion_matrix


from matplotlib.pylab import rcParams
rcParams['figure.figsize'] = 12, 4



train_file_name = 'E:/2021spr_term_class/课程——移动互联网应用设计/互移课设数据/paderborn-train/smote_feature/trainfea_resamp_new_N15_M01_F10.csv'
test_file_name = 'E:/2021spr_term_class/课程——移动互联网应用设计/互移课设数据/paderborn-train/smote_feature/trainfea_resamp_new_N15_M07_F10.csv'

data = np.array(pd.read_csv(train_file_name))
X = data[:,:-1]
y = data[:,-1].astype(int)

#X_train, X_test, y_train, y_test = train_test_split(X, y, test_size=0.3, random_state=42)
data1 = np.array(pd.read_csv(test_file_name))
X1 = data1[:,:-1]
y1 = data1[:,-1].astype(int)

#--------------------------------------------------------------
def w_score(test_y,predict):
        cfu_matr = confusion_matrix(test_y,predict)
        acu = np.array([cfu_matr[i,i]/len(test_y[test_y==i]) for i in range(len(cfu_matr))])
        w = np.array([0.1,0.2,0.25,0.2,0.25])
        print('混淆矩阵：',cfu_matr,sep='\n')
        print('每个类别的预测准确率：',acu,sep='\n')
        print('加权后的score：',np.sum(acu*w),sep='\n')


xgtrain = xgb.DMatrix(X ,label = y)
#--------------------------------------------------------------
#number One
#Choose all predictors except target & IDcols
clf = XGBClassifier(
        n_estimators=5000,    #总迭代数
        max_depth=4,    #树的深度
        min_child_weight=1, #。。。
        subsample=0.65,
        colsample_bytree=0.7,
 
        learning_rate =0.01,
        objective= 'multi:softmax',    #需要被最小化的损失函数，选的是多分类预测类别
        num_class= 5,    #指定类别数目
        gamma=0,    #惩罚参数
        reg_alpha=0.05,
        reg_lambda=0.05,
        nthread=4,
        seed=27)
xgtrain = xgb.DMatrix(X ,label = y)
xgb_param = clf.get_xgb_params()
cvresult = xgb.cv(xgb_param, xgtrain, 
                  num_boost_round = 5000, nfold = 5,
                  metrics=['mlogloss'], early_stopping_rounds=50,
                  stratified=True, seed=1301)

print('Best number of trees = {}'.format(cvresult.shape[0]))
clf.set_params(n_estimators=cvresult.shape[0],use_label_encoder=False)#把clf的参数设置成最好的树对应的参数
clf.fit(X,y,eval_metric='merror')
dtest_x = xgb.DMatrix(X1)
pre = clf.predict(X1)
w_score(y1,pre)
#--------------------------------------------------------------
#number Two
#sklearn的CV调优
param_test1 = {
 'reg_alpha':[0.01, 0.05, 0.1, 0.2, 0.5],
 'reg_lambda':[0.05,0.1,0.2,0.5]
}

clf1 = XGBClassifier( learning_rate =0.1, n_estimators=112, 
                     max_depth=4,min_child_weight=1, gamma=0.1, 
                     subsample=0.65, colsample_bytree=0.7,
                     objective= 'multi:softmax', num_class= 5,    #指定类别数目
                     nthread=4,   seed=27,
                     use_label_encoder=False)
xgtrain = xgb.DMatrix(X ,label = y)
gsearch1 = GridSearchCV(estimator = clf1, param_grid = param_test1, 
                        scoring = 'accuracy',n_jobs=4, cv=5)
gsearch1.fit(X,y,eval_metric='merror')
#print(gsearch1.grid_scores_)
print(gsearch1.best_params_) 
print(gsearch1.best_score_)