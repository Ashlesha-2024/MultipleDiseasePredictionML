# -*- coding: utf-8 -*-
"""
Created on Fri Oct 14 15:28:24 2022

@author: Ashlesha Ahirwadi
"""
import numpy as np
import pandas as pd
from sklearn.model_selection import train_test_split
from sklearn.linear_model import LogisticRegression
 #dataset
df = pd.read_csv(r'C:/Users/ashle/OneDrive/Desktop/Projects/ML-3 Multiple diseases/Heart Disease/heart_disease_data.csv')
# get the locations
X = df.iloc[:, :-1]
y = df.iloc[:, -1]
#train test split
X_train, X_test, y_train, y_test = train_test_split(
	X, y, train_size=0.8, random_state=1)
#modelling
hd_model = LogisticRegression()
hd_model.fit(X_train,y_train)
# making predictions
input_data = (43,0,2,122,213,0,1,165,0,0.2,1,0,2)

# change the input data to a numpy array
input_data_as_numpy_array= np.asarray(input_data)

# reshape the numpy array as we are predicting for only on instance
input_data_reshaped = input_data_as_numpy_array.reshape(1,-1)

prediction = hd_model.predict(input_data_reshaped)
print(prediction)

if (prediction[0]== 0):
  print('The Person does not have a Heart Disease')
else:
  print('The Person has Heart Disease')
  #Save file
import pickle
filename = 'heartdisease_model.sav'
pickle.dump(hd_model, open(filename, 'wb'))
loaded_model = pickle.load(open('heartdisease_model.sav', 'rb'))
for column in X.columns:
  print(column)  