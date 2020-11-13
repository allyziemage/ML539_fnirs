from sktime.classification.compose import BOSSEnsemble
from sktime.classification.compose import ColumnEnsembleClassifier
from sklearn.model_selection import train_test_split
from sklearn.metrics import accuracy_score

import numpy as np
import pandas as pd

from numpy import savetxt

import os
import csv

raw_df= pd.read_csv(file_name)
print(raw_df)

X, y = load_arrow_head(return_X_y=True)
X_train, X_test, y_train, y_test = train_test_split(X, y)
classifier = BOSSEnsemble()
classifier.fit(X_train, y_train)
y_pred = classifier.predict(X_test)
accuracy_score(y_test, y_pred)