# Project repoistory for CS539 - Machine Learning 

## Developed by Team 3 - Jocelyn Petitto, Shima Azizi, Kathleen Cachel, Amisha Jindal, and Alicia Howell

### Environment requirements
To run each algorithm's script, you must have the following packages: sklearn, sktime, statistics, numpy, pandas, time, sys, json, datetime, os, csv.

### General notes
Each algorithm was implemented in its own script, so there are differences in the methods to run each algorithm. Please be sure to read the algorithm's section in the README.md to understand how to use it.

There are four data files, but only three datasets. **eeg_small_uci.csv** is the small UCI EEG dataset and is used individually. **211_AXCPT19_TK_axcaxwr_hb_cs539.csv** is the WPI fNIRS dataset and is used individually. **eeg_test_uci.csv** and **eeg_train_uci.csv** belong to the large UCI EEG dataset and are run together. **eeg_test_uci.csv** is the designated testing data from the original researchers who collected the data, and **eeg_train_uci.csv** is the designated training data. They contain approximately equal amounts of data. 

### Time Series Forest Classifier (TSF)
instructions to run TSF

### MrSEQL
The code is in the form of a notebook and has different cells for the three experiments, given by their titles. In order to run a particular case, just uncomment the particular cell and comment the other cells.

### Bag of SFA Symbols Ensemble (BOSSE)
BOSSE is ran in through the terminal command line. This will cover how to do so with Windows OS, but should easily transfer to another OS. 

Navigate to the bossensemble directory from the root directory. Contained here are eleven files: two python scripts, four csv files, and five jason files. Below are detailed summaries of what each file contains, but in general, the python scripts with be called in the command line and contains the algorithm for processing the data, the jason files will be called in the command line and contain the parameters, dataset names, and other details necessary to fit the models, and the csv files will be called through the jason files. 

> bossensemble.py

This python script is the base of the boss ensemble classifier. It contains functions to read the jason and csv files, to reformat the csv in to a pandas dataframe, to call column ensemble, and to call boss ensemble from column ensemble. It can only take one csv file and thus will only work with the following jason files: **config_eeg_small.json**, **config_eeg_small_multichannel.json**, **config_fnirs.json**, **config_fnirs_multichannel.json**. 

### K Nearest Neighbors (KNN)
instructions to run KNN

### Proximity Forest (PF)
instructions to...run? PF :D
