# -*- coding: utf-8 -*-
from sklearn.model_selection import train_test_split

from sktime.classification.compose import (
    ColumnEnsembleClassifier,
    TimeSeriesForestClassifier,
)
import pandas as pd
from sklearn import preprocessing
import time
import matplotlib.pyplot as plt
# ==============TimeSeriesForest on fNIRS Data========================================
namesCol = ['Subject Name', 'Event Name', 'Channel Name', 'Start time', 'End time']

for q in range(131):
  strVar = 'v' + str(q)
  namesCol.append(strVar)

df = pd.read_csv('211_AXCPT19_TK_axcaxwr_hb_cs539.csv', header = 0, names = namesCol)

col_name = list(df.columns)
trans_df = pd.DataFrame(data = df, columns = col_name)


# Get Y-target -df
y = trans_df['Event Name']

# Drop target variable and get X-feature- df
X = trans_df.drop(['Event Name'], axis = 1)

# Splitting the dataset: 
#random_state simply sets a seed to the random generator, so that your train-test splits are always deterministic. If you don't set a seed, it is different each time.
X_train, X_test, y_train, y_test = train_test_split(X, y, test_size=0.4, random_state=42)#test_size specifies percetage of split between test and train
#print(X_train.shape, y_train.shape, X_test.shape, y_test.shape)
#print(X_train.head())

cat_cols = ['Subject Name', 'Channel Name']
enc = preprocessing.LabelEncoder()

for col in cat_cols:
    X_train[col] = X_train[col].astype('str')
    X_test[col] = X_test[col].astype('str')
    X_train[col] = enc.fit_transform(X_train[col])
    X_test[col] = enc.transform(X_test[col])


#isolate the time series
X_train_timedata = X_train[X_train.columns[4:136]]
X_test_timedata = X_test[X_test.columns[4:136]]


# Conversion to numpy array
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

# =======================Column ensembling================================ 
Num_Estimator_List =[10, 20, 30, 40, 50, 60, 70, 80, 90, 100]

Ac =[] #Accuracy
Ef =[] #Efficiency
for n in Num_Estimator_List:
    clf = ColumnEnsembleClassifier(
        estimators=[
            ("TSF0", TimeSeriesForestClassifier(n_estimators=n), [0]),
        ]
    )

    start_time= time.time()
    clf.fit(X_ts_train, y_train)
    Efficiency= time.time() - start_time
    Ef.append(Efficiency)

    Accuracy = clf.score(X_ts_test, y_test)
    Ac.append(Accuracy)
    
print("Efficiency is:\n", Ef)
print("Accuracy is :\n",Ac)


fig,axes=plt.subplots() 
plt.plot(Num_Estimator_List, Ac, color="darkgreen", marker='o',markerfacecolor='mediumvioletred', markersize=6,linewidth=2, alpha=0.9,linestyle='--', label="Accuracy")     
plt.title("60-40 Train-Test Split")      
plt.xlabel("Number of Estimators")
plt.ylabel("Accuracy")

fig,axes=plt.subplots() 
plt.plot(Num_Estimator_List, Ef, color="steelblue", marker='o',markerfacecolor='mediumvioletred', markersize=6,linewidth=2, alpha=0.9,linestyle='--', label="Efficiency")     
plt.title("60-40 Train-Test Split")      
plt.xlabel("Number of Estimators")
plt.ylabel("Efficiency (Seconds)")