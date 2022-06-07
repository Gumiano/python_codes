# -*- coding: utf-8 -*-

# 代码10-8

import pandas as pd
from sklearn.preprocessing import StandardScaler
from sklearn.neural_network import MLPClassifier
from sklearn.externals import joblib

# 读取数据
Xtrain = pd.read_excel('../tmp/sj_final.xlsx')
ytrain = pd.read_excel('../data/water_heater_log.xlsx')
test = pd.read_excel('../data/test_data.xlsx')
# 训练集测试集区分。
x_train, x_test, y_train, y_test = Xtrain.iloc[:,5:],test.iloc[:,4:-1],\
                                   ytrain.iloc[:,-1],test.iloc[:,-1]
# 标准化
stdScaler = StandardScaler().fit(x_train)
x_stdtrain = stdScaler.transform(x_train)
x_stdtest = stdScaler.transform(x_test)
# 建立模型
bpnn = MLPClassifier(hidden_layer_sizes = (17,10), max_iter = 200, solver = 'lbfgs',random_state=50)
bpnn.fit(x_stdtrain, y_train)
# 保存模型
joblib.dump(bpnn,'../tmp/water_heater_nnet.m')
print('构建的模型为：\n',bpnn)
