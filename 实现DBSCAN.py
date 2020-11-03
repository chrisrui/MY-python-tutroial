import pandas as pd

from pandas.plotting import scatter_matrix
data_train=pd.read_excel('C:/Users/777/Desktop/加班/日报当月数据.xlsx',sheet_name='当月数据数据')
data_train.head(5)

from sklearn.cluster import DBSCAN
data_train.columns

x=data_train[['结算金额','结算人数','费率']]
x
x.count()
db_train=DBSCAN(eps=10,min_samples=2).fit(x)

labels=db_train.labels_
labels
x.columns
x.loc[:,'dbscan']=labels
x=x.sort_values('dbscan')
x
d1=x.groupby('dbscan').mean()
type(d1)
d1.count()

from sklearn import  metrics
score=metrics.silhouette_score(x,labels)

score
