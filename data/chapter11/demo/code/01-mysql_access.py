# -*- coding: utf-8 -*-

# 代码11-1

import os
import pandas as pd


# 修改工作路径到指定文件夹
os.chdir("D:/chapter11/demo")

# 第一种连接方式
from sqlalchemy import create_engine

engine = create_engine('mysql+pymysql://root:123@192.168.31.140:3306/test?charset=utf8')
sql = pd.read_sql('all_gzdata', engine, chunksize = 10000)

# 第二种连接方式
import pymysql as pm

con = pm.connect('localhost','root','123456','test',charset='utf8')
data = pd.read_sql('select * from all_gzdata',con=con)
con.close()           #关闭连接

# 保存读取的数据
data.to_csv('./tmp/all_gzdata.csv', index=False, encoding='utf-8')

