# Project repoistory for CS539 - Machine Learning 

## Developed by Team 3 - Jocelyn Petitto, Shima Azizi, Kathleen Cachel, Amisha Jindal, and Alicia Howell

### Environment requirements
To run each algorithm's script, you must have the following packages: sklearn, sktime, statistics, numpy, pandas, time, sys, json, datetime, os, csv.

### General notes
Each algorithm was implemented in its own script, so there are differences in the methods to run each algorithm. Please be sure to read the algorithm's section in the README.md to understand how to use it.

There are four data files, but only three datasets. **eeg_small_uci.csv** is the small UCI EEG dataset and is used individually. **211_AXCPT19_TK_axcaxwr_hb_cs539.csv** is the WPI fNIRS dataset and is used individually. **eeg_test_uci.csv** and **eeg_train_uci.csv** belong to the large UCI EEG dataset and are run together. **eeg_test_uci.csv** is the designated testing data from the original researchers who collected the data, and **eeg_train_uci.csv** is the designated training data. They contain approximately equal amounts of data. 

### Time Series Forest Classifier (TSF)
There are three different .py file in the TSF folder including **TimeSeriesForest_fNIRS.py**, **TimeSeriesForest_EEG_Small.py**, and **TimeSeriesForest_EEG_Large.py** running the TSF algorithm on fNIRS dataset, EEG small dataset, and EEG large dataset, respectively. We evaluate the accuracy and efficency for varying number of estimators (number of trees) and plot the results.

### MrSEQL
The code is in the form of a notebook and has different cells for the three experiments, given by their titles. In order to run a particular case, just uncomment the particular cell and comment the other cells.

### Bag of SFA Symbols Ensemble (BOSSE)
BOSSE is ran through the terminal command line. This will cover how to do so with Windows OS, but should easily transfer to another OS. 

#### File descriptions
Navigate to the bossensemble directory from the root directory. Contained here are eleven files: two python scripts, four csv files, and five jason files. Below are detailed summaries of what each file contains, but in general, the python scripts with be called in the command line and contains the algorithm for processing the data, the jason files will be called in the command line and contain the parameters, dataset names, and other details necessary to fit the models, and the csv files will be called through the jason files. 

> bossensemble.py

This python script is the base of the boss ensemble classifier. It contains functions to read the jason and csv files, to reformat the csv in to a pandas dataframe, to call column ensemble, and to call boss ensemble from column ensemble. It can only take one csv file and thus will only work with the following jason files: **config_eeg_small.json**, **config_eeg_small_multichannel.json**, **config_fnirs.json**, **config_fnirs_multichannel.json**. This script will perform a randomized split of the data into training and testing sets. The split is determined by user input into the jason configuration file. Common inputs are 0.5, 0.6, 0.7, and 0.8, which designate the amount of data in the training set. 

> bossensemble_splitfile.py

This script is highly similar to **bossensemble.py** with the main difference being that it takes two csv files, and thus will only work with **config_split.json** as the configuration file. This script does not perform a randomized separation of training and testing data. Instead, it takes two csv files, one which is the training set and one which is the testing set. In all other regards, it runs the same as **bossensemble.py**.

> config_eeg_small.json

This is one of the jason configuration files. It contains variables for the file path of the csv file, the class name, the training percentage, and a list of jobs to be run with the preceding variables. For this file, the path for the small UCI EEG dataset is determined and we set the algorithm to classify based on the "event" feature. Please do not modify those two variables. The training percentage is set to 50%, but this can be modified, along with the jobs. Currently, the file is set to run through all of the parameters that were tested in our experiments for BOSSE. Due to the runtime for the BOSSE algorithm, this config file only calls one out of the sixty-four channels in the EEG dataset. 

> config_eeg_small_multichannel.json

This is another configuration file. It contains the same variables as the proceeding config file and is also calling the small UCI EEG dataset and classifying on the "event" feature. However, this config file runs jobs on ten randomly selected channels from the sixty-four available channels. Each parameter is consistent across all channels and are not the sktime defaults. Please tune there parameters if so desired. The training percentage is at 0.8, but this can be modified. 

> config_fnirs.json

This configuration file has the same variables as the proceeding files, except it is now calling the WPI fNIRS dataset. It contains the same tuning parameters as **config_eeg_small.json** and implements one out of the twenty available channels in the fNIRS dataset. The training percentage is set to 0.7, but can be modified. 

> config_fnirs_multichannel.json

Similar to the **config_eeg_small_multichannel.json** file, this takes the same variables as the proceeding config files and contains ten randomly selected channels from the twenty available channels. The training percentage is set to 0.5, but can be modified. The parameters in the jobs can be varied and are currently the same as those in **config_eeg_small_multichannel.json**, which are not the sktime defaults. 

> config_split.json

This is the only provided configuration file that can be ran on **bossensemble_splitfile.py**. It contains variables for training file path (filePath), testing file path (testPath), the class name, and jobs. It does not contain a training percentage because the training and testing sets are predefined. Currently, it takes in the sktime default parameters and only one channel out of the sixty-four available channels. We caution adding more channels, because this will significantly increase the run time of the classifier by the magnitude of hours. 

> 211_AXCPT19_TK_axcaxwr_hb_cs539.csv

This is one of the available csvs to run with **bossensemble.py**. It contains the fNIRS dataset from WPI. More details about the data structure can be viewed on our Google Sites website. 

> eeg_small_uci.csv

This is the other csv that can be ran with **bossensemble.py**. It contains the small UCI EEG dataset from UCI Archive. More details about the data structure can be viewed on our Google Sites website. 

> eeg_test_uci.csv
> eeg_train_uci.csv

These two datasets must be used together with **bossensemble_splitfile.py**. As their file names denote, one contains a predefined testing set and the other contains the predefined training set, as detailed on UCI Archive. More details about the data structure can be viewed on our Google Sites website. 

#### How to run
To run either script, navigate to the bossensemble directory within the root directory in a terminal window, such as PowerShell. This can be done with:

>cd userpath/ML539_fnirs/bossensemble

Next, choose a script (either **bossensemble.py** or **bossensemble_splitfile.py**) and a correlating configuration file. If you are unsure which configuration file works for each script, please view the file notes above. Next, run the chosen script and configuration file in the command line, with specific ordering of script first, then configuration file. Below is an example that would be acceptable in PowerShell:

>python bossensemble.py config_fnirs.json

There will be a variety of outputs in the terminal window to update the user as the script runs. These include specifying what file was inputted and it's related variables, when the data has finished reformatting, and the results from the model. Time results will be given in seconds (s). 

### K Nearest Neighbors (KNN)
There are three seperate notebooks for KNN. **KNN_eegLargenotebook.ipynb** imports imports the **eeg_test_uci.csv** and **eeg_train_uci.csv** dataset. **KNN_eegsmallnotebook.ipynb** imports imports the **eeg_small_uci.csv**.**KNN_fnirs_notebook.ipynb** imports imports the **fnirs.csv**, which is the **211_AXCPT19_TK_axcaxwr_hb_cs539.csv** renamed. The different features and their corresponding names, splitting of test and train for the small eeg data and fnirs data, and pre-seperation of training and test sets for the large eeg data are done in the specific notebooks. 
The notebooks were run in the Google Colab environment, and the filepath for the specific csvs assumes the csvs have been loaded into the current colab session. To run the notebook import the relevent csv to the session and run the cells. The bulk of the code prior to fitting KNN is manipulating the data into the proper format for the classifier (pd.Series). Included are cells contianing code to create accuracy versus k hyperparameter plots. Those can be skipped if desired. Changing the `num_N` variable changes the `n_neighbors` parameter. The distance metric can also by changed by changing the `metric` parameter. It is currently set to dynamic time warping for best results. 

### Proximity Forest (PF)
There are two seperate notebooks for ProximityForest. **ML_FP_ProximityForestPlayground.ipynb** imports the **eeg_test_uci.csv** and **eeg_train_uci.csv** datasets and **ML_FP_ProximityForestPlayground_fNIRS.ipynb** imports **211_AXCPT19_TK_axcaxwr_hb_cs539.csv**. The difference in features and feature names in the datasets as well as the pre-seperated training and test sets for the eeg data and the need to split the data into training and test sets for the fNIRS dataset are accounted for in the individual notebooks.
The notebooks included were run in the Google Colab environment, so the current filepath points at files on the Drive for the corresponding account. To run either notebook, confirm the path to the corresponding * *.csv* is correct as it is stored on your system.
Varying the `n_estimators` and `max_depth` parameters when instantiating the class will allow reproduction of the conditions under which the **211_AXCPT19_TK_axcaxwr_hb_cs539.csv** dataset was used to train and test a variety of Proximity Forest models.
There are known issues installing sktime on Mac that specifically affect the ability to model using Proximity Forest. Alternative installation instructions are included in the sktime documentation, but they were not found to work across all systems.
