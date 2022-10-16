# -*- coding: utf-8 -*-
"""
Created on Fri Oct 14 15:52:18 2022

@author: ashle
"""

# Importing libararies
import numpy as np #Usefull in processing
import pandas as pd #Usefull in structuring our dataset
from sklearn.preprocessing import StandardScaler # use to standardise data
from sklearn.model_selection import train_test_split
from sklearn import svm


#Data 
df = pd.read_csv(r'C:/Users/ashle/OneDrive/Desktop/Projects/ML-3 Multiple diseases/Parkinson/parkinsons.csv')
# get the locations
#X = df.drop(columns=['name','status'])
X = df.iloc[:, :-1]
y = df.iloc[:, -1]

# split the dataset
X_train, X_test, y_train, y_test = train_test_split(
	X, y, train_size=0.8, random_state=1)

#Data Standardisation
#To help data make better predictions
scaler = StandardScaler()
scaler.fit(X_train)
X_train = scaler.transform(X_train)

X_test = scaler.transform(X_test)

#train model
p_model=svm.SVC(kernel='linear')
p_model.fit(X_train,y_train)

#Make Prediction
input_data = (202.266,211.604,197.079,0.0018,0.000009,0.00093,0.00107,0.00278,0.00954,0.085,0.00469,0.00606,0.00719)

# changing the input_data to numpy array
input_data_as_numpy_array = np.asarray(input_data)

# reshape the array as we are predicting for one instance
input_data_reshaped = input_data_as_numpy_array.reshape(1,-1)

# standardize the input data
std_data = scaler.transform(input_data_reshaped)

prediction = p_model.predict(std_data)
print(prediction)

if (prediction[0] == 0):
  print('The person does not have parkinson')
else:
  print('The person has parkinson')
  
  #Save file
import pickle
filename = 'parkinsons_model.sav'
pickle.dump(p_model, open(filename, 'wb'))
loaded_model = pickle.load(open('parkinsons_model.sav', 'rb'))
for column in X.columns:
  print(column)