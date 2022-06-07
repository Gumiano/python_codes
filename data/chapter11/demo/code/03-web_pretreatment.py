# -*- coding: utf-8 -*-

# 代码11-10

import os
import re
import pandas as pd
import pymysql as pm
from random import sample

# 修改工作路径到指定文件夹
os.chdir("D:/chapter11/demo")

# 读取数据
con = pm.connect('localhost','root','123456','test',charset='utf8')
data = pd.read_sql('select * from all_gzdata',con=con)
con.close()  # 关闭连接

# 取出107类型数据
index107 = [re.search('107',str(i))!=None for i in data.loc[:,'fullURLId']]
data_107 = data.loc[index107,:]

# 在107类型中筛选出婚姻类数据
index = [re.search('hunyin',str(i))!=None for i in data_107.loc[:,'fullURL']]
data_hunyin = data_107.loc[index,:]

# 提取所需字段(realIP、fullURL)
info = data_hunyin.loc[:,['realIP','fullURL']]

# 去除网址中“？”及其后面内容
da = [re.sub('\?.*','',str(i)) for i in info.loc[:,'fullURL']]
info.loc[:,'fullURL'] = da     # 将info中‘fullURL’那列换成da
# 去除无html网址
index = [re.search('\.html',str(i))!=None for i in info.loc[:,'fullURL']]
index.count(True)   # True 或者 1 ， False 或者 0
info1 = info.loc[index,:]



# 代码11-11

# 找出翻页和非翻页网址
index = [re.search('/\d+_\d+\.html',i)!=None for i in info1.loc[:,'fullURL']]
index1 = [i==False for i in index]
info1_1 = info1.loc[index,:]   # 带翻页网址
info1_2 = info1.loc[index1,:]  # 无翻页网址
# 将翻页网址还原
da = [re.sub('_\d+\.html','.html',str(i)) for i in info1_1.loc[:,'fullURL']]
info1_1.loc[:,'fullURL'] = da
# 翻页与非翻页网址合并
frames = [info1_1,info1_2]
info2 = pd.concat(frames)
# 或者
info2 = pd.concat([info1_1,info1_2],axis = 0)   # 默认为0，即行合并
# 去重（realIP和fullURL两列相同）
info3 = info2.drop_duplicates()
# 将IP转换成字符型数据
info3.iloc[:,0] = [str(index) for index in info3.iloc[:,0]]
info3.iloc[:,1] = [str(index) for index in info3.iloc[:,1]]
len(info3)



# 代码11-12

# 筛选满足一定浏览次数的IP
IP_count = info3['realIP'].value_counts()
# 找出IP集合
IP = list(IP_count.index)
count = list(IP_count.values)
# 统计每个IP的浏览次数，并存放进IP_count数据框中,第一列为IP，第二列为浏览次数
IP_count = pd.DataFrame({'IP':IP,'count':count})
# 3.3筛选出浏览网址在n次以上的IP集合
n = 2
index = IP_count.loc[:,'count']>n
IP_index = IP_count.loc[index,'IP']



# 代码11-13

# 划分IP集合为训练集和测试集
index_tr = sample(range(0,len(IP_index)),int(len(IP_index)*0.8))  # 或者np.random.sample
index_te = [i for i in range(0,len(IP_index)) if i not in index_tr]
IP_tr = IP_index[index_tr]
IP_te = IP_index[index_te]
# 将对应数据集划分为训练集和测试集
index_tr = [i in list(IP_tr) for i in info3.loc[:,'realIP']]
index_te = [i in list(IP_te) for i in info3.loc[:,'realIP']]
data_tr = info3.loc[index_tr,:]
data_te = info3.loc[index_te,:]
print(len(data_tr))
IP_tr = data_tr.iloc[:,0]  # 训练集IP
url_tr = data_tr.iloc[:,1]  # 训练集网址
IP_tr = list(set(IP_tr))  # 去重处理
url_tr = list(set(url_tr))  # 去重处理
len(url_tr)
