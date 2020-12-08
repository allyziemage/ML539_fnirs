from sklearn.pipeline import Pipeline
from sklearn.model_selection import train_test_split
from sklearn.model_selection import KFold

from statistics import mean

from sktime.classification.dictionary_based import BOSSEnsemble
from sktime.classification.compose import ColumnEnsembleClassifier

import numpy as np
import pandas as pd

from numpy import savetxt

# cleaning up whatever happened in loading data
# feed in the data
from sktime.utils.load_data import from_long_to_nested
import time
import sys
import json
from datetime import date
from datetime import datetime

import os
import csv


#setting constants

classifier={
    'BOSSE_CLF' : 0,
}

BOSSE_default_parameters={'min_window':10,'threshold':0.92}

# converts a list to a dictionary
#ex: input ["john",1,"kate",3]-> {"john":1, "kate":3}
def list_to_dict(a):
    it = iter(a)
    res_dct = dict(zip(it, it))
    return res_dct

#extracting the data from the csv as a dataframe and reformatting it for sktime compatibility
def reformatData(target, file_name):

    print("reformatting the data...")
    raw_df= pd.read_csv(file_name)

    # find count of events in the datasets
    events_count = raw_df['event'].value_counts().to_dict()


    #collapses the time cols into one single time column to match the rest of the columns
    long_table_df= raw_df.melt(id_vars=["event", "name","start time", "end time","channel"],
            var_name="anindex",
            value_name="value")

    sorted_long_table_df=long_table_df.sort_values(by=['event','name','start time','channel'], axis=0)

    # combine start time and subject into 1 column
    sorted_long_table_df['case_key'] = sorted_long_table_df['start time'].astype(str) + sorted_long_table_df['name']

    # use column case_key to be unique index for reformatted table
    unique_dim_ids = sorted_long_table_df.iloc[:, 4].unique()

    # get a mapping of channels
    channels_map = {}
    #replacing channel named to numeric values (need to do this doem the from_long_to_nested function)
    for i in range(len(unique_dim_ids)):
        my_channel=unique_dim_ids[i]
        sorted_long_table_df['channel']=sorted_long_table_df['channel'].replace({my_channel:i})
        channels_map[i]=my_channel

    unique_case_key = sorted_long_table_df.iloc[:, -1].unique()
    for i in range(len(unique_case_key)):
        my_case_key=unique_case_key[i]
        sorted_long_table_df['case_key']=sorted_long_table_df['case_key'].replace({my_case_key:i})


    # might need to delete this check if it takes too long
    time_map={} # a map index for time
    unique_start_time = sorted_long_table_df.iloc[:, 2].unique()
    for i in range(len(unique_start_time)):
        my_time=unique_start_time[i]
        sorted_long_table_df['start time']=sorted_long_table_df['start time'].replace({my_time:i})
        time_map[i]= my_time

    #excess columns are dropped for the frome_long_to_nested function
    sorted_long_table_df_stripped=sorted_long_table_df.drop(columns=['event','name','end time','start time'])

    # reorder column, move case_key column from last to first
    sorted_long_table_df_stripped = sorted_long_table_df_stripped[['case_key','channel','anindex','value']]
    #table goes from long to nested
     # returns a sktime-formatted dataset with individual dimensions represented by columns of the output dataframe:
    df_nested = from_long_to_nested(sorted_long_table_df_stripped)

    # create a list of labels
    new_unique_case_key=sorted_long_table_df.iloc[:, -1].unique()
    labels=[]
    for e in new_unique_case_key:
        x=sorted_long_table_df.loc[sorted_long_table_df['case_key']==e,[target]].iloc[0][0]
        labels.append(x)

    np_labels= np.asarray(labels, dtype=np.str)

    return df_nested, np_labels, events_count

#receives a classifier name and a list of parameters
#outputs specified classifier
def classifierBuilder(clf_name,params):
    clf_params_dict=list_to_dict(params)
    if(clf_name == 'BOSSE_CLF'):
        BOSSE_params=BOSSE_default_parameters
        for e in clf_params_dict:
            BOSSE_params[e]=clf_params_dict[e]
        clf = BOSSEnsemble(min_window=BOSSE_params['min_window'], threshold=BOSSE_params['threshold'])
    else:
        raise ValueError("Specified classifier is not an option")
    return clf

# splitting the test train based on the test train that is specified
def splitTestTrain(X, y, percent_train):
    Xtrain, Xtest, ytrain, ytest = train_test_split(X, y, train_size = percent_train)
    return Xtrain, Xtest, ytrain, ytest

#creates an ensemble of several classifier
#outputs a classification score
def columnEnsembleMethod(classifier_list,X,y,percent_train,clf_parameters=[]):
    #generate tuples (and format accordingly for the ensembler)
    estimator_list=[]
    Xtrain, Xtest, ytrain, ytest= splitTestTrain(X,y,percent_train)
    for i in classifier_list:
        params=[]
        built_clf = classifierBuilder(i['classifier'], params)
        num = i['columnNum']
        name = i['classifier']+str(num)
        estimator_list.append((name,built_clf,[num]))
    clf = ColumnEnsembleClassifier(estimators=estimator_list)
    start_time=time.time()
    clf.fit(Xtrain, ytrain)
    end_time= time.time() - start_time
    print('Total Time : '+str(round(end_time,2))+' seconds\n\n')
    return clf.score(Xtest, ytest)


'''check if the headers of the files are correct yet'''
def check_headers(csv_filepath):
    with open(csv_filepath, 'r') as csv_file:
        csv_reader = csv.reader(csv_file, delimiter=',')
        headers = next(csv_reader)
    return headers[0]=='name'

def update_headers(csv_filepath):
    # get the csv file from json
    with open(csv_filepath, 'r') as csv_file:
        csv_reader = csv.reader(csv_file, delimiter=',')
        headers = next(csv_reader)  # Compatible with Python 3.x (also 2.7)
        headers=['name', 'event', 'channel', 'start time', 'end time']
        second =next(csv_reader)
        n = len(headers)
        for row in csv_reader:
            n=max(n, len(row))
        for i in range(5, n):
            headers.append(i-4)

    with open(csv_filepath, 'r') as fp:
        reader = csv.DictReader(fp, fieldnames=headers)
        # use newline='' to avoid adding new CR at end of line
        new_csv_filename= '/'.join(csv_filepath.split('/')[:-1])+'/'+csv_filepath.split('/')[-1].split('.')[0]+'_updated.csv'
        print(new_csv_filename)
        with open(new_csv_filename, 'w', newline='') as fh:
            writer = csv.DictWriter(fh, fieldnames=reader.fieldnames)
            writer.writeheader()
            header_mapping = next(reader)
            writer.writerows(reader)
    return new_csv_filename

def main():
    script = sys.argv[0]
    json_file_name = sys.argv[1]

    #reading the inputted json file
    print(json_file_name)

    with open(json_file_name) as f:
        data = json.load(f)

    print(data)
    # check if the headers are correct
    if not check_headers(data['filePath']):
        # adding small scripts to fix the headers if wrong format
        data['filePath'] = update_headers(data['filePath'])

    print(json.dumps(data, indent=4, sort_keys=True))

    #setting values given the configuration files
    target=data['targetCol']
    file_name=data['filePath']
    print(file_name)
    percent_train=data['percentTrain']

    X, y, events_count =reformatData(target,file_name)

    for job in data['jobs']:
        acc= 0
        print("JOB:")
        print(job)
        params=[]
        if ('parameters' in job):
            params=job['parameters']
        if(job['method']=='COLUMN_ENSEMBLE'):
            acc=columnEnsembleMethod(job['ensembleInfo'],X,y,percent_train,clf_parameters=params)
        else:
            raise ValueError(str(job['method']) +" classification method does not exist")
        print("Accuracy : "+str(round(acc*100,2))+'%\n')

main()