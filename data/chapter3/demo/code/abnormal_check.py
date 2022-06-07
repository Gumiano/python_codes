import pandas as pd

data = pd.read_excel('../data/catering_sale.xls')
print(data.describe())