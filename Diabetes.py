# -*- coding: utf-8 -*-
"""
Created on Thu Oct 13 19:57:48 2022

@author: Ashlesha Ahirwadi
"""
# Importing libararies
import numpy as np #Usefull in processing
import pandas as pd #Usefull in structuring our dataset
from sklearn.preprocessing import StandardScaler # use to standardise data
from sklearn.model_selection import train_test_split
from sklearn import svm


#Data 
df = pd.read_csv(r'C:/Users/ashle/OneDrive/Desktop/Projects/ML-3 Multiple diseases/Diabetes/diabetes.csv')
# get the locations
X = df.iloc[:, :-1]
y = df.iloc[:, -1]

#Data Standardisation
#To help data make better predictions
scaler = StandardScaler()
scaler.fit(X)
standardized_data = scaler.transform(X)
#X = standardized_data
#Y = df['Outcome']
# split the dataset
X_train, X_test, y_train, y_test = train_test_split(
	X, y, train_size=0.8, random_state=1)

#train model
svm1=svm.SVC(kernel='linear')
svm1.fit(X_train,y_train)

#Make Prediction
input_data = (1,103,30,38,83,43.3,0.183,33)

# changing the input_data to numpy array
input_data_as_numpy_array = np.asarray(input_data)

# reshape the array as we are predicting for one instance
input_data_reshaped = input_data_as_numpy_array.reshape(1,-1)

# standardize the input data
std_data = scaler.transform(input_data_reshaped)

prediction = svm1.predict(std_data)
print(prediction)

if (prediction[0] == 0):
  print('The person is not diabetic')
else:
  print('The person is diabetic')

  #Save file
import pickle
filename = 'diabetes_model.sav'
pickle.dump(svm1, open(filename, 'wb'))
loaded_model = pickle.load(open('diabetes_model.sav', 'rb'))
for column in X.columns:
  print(column)