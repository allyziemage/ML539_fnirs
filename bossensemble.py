from sktime.classification.dictionary_based import BOSSEnsemble
from sktime.classification.compose import ColumnEnsembleClassifier
from sklearn.model_selection import train_test_split
from sklearn.metrics import accuracy_score

import numpy as np
import pandas as pd

from numpy import savetxt

import os
import csv
#Load the data file
def main():
	filename = input("Please enter the file name, inlcuding path and extension: ")
	
	#get fnirs data from csv
	file_data = np.genfromtxt(filename, delimiter=',')
	column_names = file_data[1,:]
	file_data = np.delete(file_data, 0, axis=0)
	inlen = list(range(0, file_data.shape[0], 1))
	fnirs_df = pd.DataFrame(index=[inlen], columns=['Data'])
	countdata = 0
	file_data = np.delete(file_data, 4, axis=1)
	file_data = np.delete(file_data, 3, axis=1)
	file_data = np.delete(file_data, 2, axis=1)
	file_data = np.delete(file_data, 0, axis=1)
	for fileline in file_data:
		empty = []
		for i in range(len(fileline)):
			empty.append(fileline[i])
		fnirs_df['Data'].iloc[countdata] = empty
		countdata = countdata + 1
	
	#get class names from csv
	with open(filename, "r") as csv_file:
		csv_reader = csv.reader(csv_file, delimiter=',')
		class_names = np.empty((0,1))
		count = 0
		for lines in csv_reader:
			if count == 0:
				count = count + 1
				continue
			else:
				class_names = np.append(class_names, lines[0])		
>>>>>>> Stashed changes

	X, y = fnirs_df, class_names
	X_train, X_test, y_train, y_test = train_test_split(X, y)
	classifier = BOSSEnsemble()
	classifier.fit(X_train, y_train)
	y_pred = classifier.predict(X_test)
	acc = accuracy_score(y_test, y_pred)
	print(acc)

main()