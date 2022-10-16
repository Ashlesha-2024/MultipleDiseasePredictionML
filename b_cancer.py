# -*- coding: utf-8 -*-
"""
Created on Fri Oct 14 22:04:48 2022

@author: Ashllesha Ahirwadi
"""

import numpy as np
import pandas as pd
import sklearn.datasets
from sklearn.model_selection import train_test_split
from sklearn.linear_model import LogisticRegression
from sklearn.metrics import accuracy_score

df = pd.read_csv(r'C:/Users/ashle/OneDrive/Desktop/Projects/ML-3 Multiple diseases/Breast Cancer/data.csv')
# get the locations
X = df.iloc[:, :-1]
y = df.iloc[:, -1]
#train test split
X_train, X_test, y_train, y_test = train_test_split(
	X, y, train_size=0.8, random_state=1)
#modelling
bc_model = LogisticRegression()
bc_model.fit(X_train,y_train)
# making predictions
input_data = (17.99,10.38,122.8,1001,0.1184,0.2776,0.3001,0.1471,0.2419,0.07871)
print(input_data)
# change the input data to a numpy array
input_data_as_numpy_array= np.asarray(input_data)

# reshape the numpy array as we are predicting for only on instance
input_data_reshaped = input_data_as_numpy_array.reshape(1,-1)

prediction = bc_model.predict(input_data_reshaped)
print(prediction)

if (prediction[0]=='M'):
  print('The Person has Malignant Tumor')
else:
  print('The Person has Benign Tumor')
  #Save file
import pickle
filename = 'breastcancer_model.sav'
pickle.dump(bc_model, open(filename, 'wb'))
loaded_model = pickle.load(open('breastcancer_model.sav', 'rb'))
for column in X.columns:
  print(column)  