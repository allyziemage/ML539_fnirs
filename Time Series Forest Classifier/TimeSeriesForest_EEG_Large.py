# -*- coding: utf-8 -*-
import numpy as np
from sklearn.model_selection import train_test_split

from sktime.classification.compose import (
    ColumnEnsembleClassifier,
    TimeSeriesForestClassifier,
)

import pandas as pd
from sklearn import preprocessing
import time

# ==========================UCI Big Dataset==========================

namesCol = ['subject', 'group', 'condition', 'trial', 'channel']

for q in range(256):
  strVar = 'v' + str(q)
  namesCol.append(strVar)

df = pd.read_csv('eeg_train_uci.csv', header = 0, names = namesCol)
col_name = list(df.columns)
trans_df = pd.DataFrame(data = df, columns = col_name)

# Get Y-target -df
y_train = trans_df['condition']

# Drop target variable and get X-feature- df
X_train = trans_df.drop(['condition'], axis = 1)

#read test data
namesCol = ['subject', 'group', 'condition', 'trial', 'channel']

for q in range(256):
  strVar = 'v' + str(q)
  namesCol.append(strVar)

df_test = pd.read_csv('eeg_test_uci.csv', header = 0, names = namesCol)
col_name = list(df.columns)
trans_df_test = pd.DataFrame(data = df, columns = col_name)

#test
# Get Y-target -df
y_test = trans_df_test['condition']

# Drop target variable and get X-feature- df
X_test = trans_df_test.drop(['condition'], axis = 1)


cat_cols = ['subject', 'group', 'trial']
enc = preprocessing.LabelEncoder()

for col in cat_cols:
    X_train[col] = X_train[col].astype('str')
    X_test[col] = X_test[col].astype('str')
    X_train[col] = enc.fit_transform(X_train[col])
    X_test[col] = enc.transform(X_test[col])

#isolate the time series
X_train_timedata = X_train[X_train.columns[4:260]]
X_test_timedata = X_test[X_test.columns[4:260]]



X_train_timedata['combine'] = X_train_timedata.values.tolist()
X_test_timedata['combine'] = X_test_timedata.values.tolist()
X_train_timedata = X_train_timedata['combine']
X_test_timedata = X_test_timedata['combine']

#convert to dataframe
X_train_timedata = X_train_timedata.to_frame()
X_test_timedata = X_test_timedata.to_frame()

ts_train = pd.Series(X_train_timedata['combine'].values, index=X_train_timedata.index)
X_ts_train = ts_train.to_frame()

ts_test = pd.Series(X_test_timedata['combine'].values, index=X_test_timedata.index)
X_ts_test = ts_test.to_frame()

for row_num in range(0,X_ts_train.shape[0]):
  series1 = pd.Series(X_ts_train.iat[row_num,0])
  X_ts_train.iat[row_num,0] = series1

for row_num in range(0,X_ts_test.shape[0]):
  series2 = pd.Series(X_ts_test.iat[row_num,0])
  X_ts_test.iat[row_num,0] = series2
  
## =======================Column ensembling================================ 
clf = ColumnEnsembleClassifier(
    estimators=[
        ("TSF0", TimeSeriesForestClassifier(n_estimators=5), [0]),
    ]
)

start_time= time.time()
clf.fit(X_ts_train, y_train)
Efficiency= time.time() - start_time
Accuracy = clf.score(X_ts_test, y_test)    
print("Efficiency is:\n", Efficiency)
print("Accuracy is :\n",Accuracy)

