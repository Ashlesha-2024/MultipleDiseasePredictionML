# -*- coding: utf-8 -*-
"""
Created on Sat Oct 15 11:15:20 2022

@author: ashle
"""

import pickle
import streamlit as st
from streamlit_option_menu import option_menu

diabetes_model = pickle.load(open('C:/Users/ashle/OneDrive/Desktop/Projects/ML-3 Multiple diseases/Diabetes/diabetes_model.sav','rb'))
heartd_model = pickle.load(open('C:/Users/ashle/OneDrive/Desktop/Projects/ML-3 Multiple diseases/Heart Disease/heartdisease_model.sav','rb'))
breastc_model = pickle.load(open('C:/Users/ashle/OneDrive/Desktop/Projects/ML-3 Multiple diseases/Breast Cancer/breastcancer_model.sav','rb'))
parki_model = pickle.load(open('C:/Users/ashle/OneDrive/Desktop/Projects/ML-3 Multiple diseases/Parkinson/parkinsons_model.sav','rb'))

#side bar
with st.sidebar:
    selected = option_menu('Multiple Disease Prediction System', ['Diabetes Prediction','Heart Disease Prediction', 'Parkinsons Prediction','Breast Cancer Prediction'],icons=['activity','heart','person','gender-female'], default_index=0)

#Diabetes Prediction
if (selected == 'Diabetes Prediction'):
    
    # page title
    st.title('Diabetes Prediction using Machine Learning')
    
    
    # getting the input data from the user
    col1, col2, col3 = st.columns(3)
    
    with col1:
        Pregnancies = st.text_input('Number of Pregnancies')
        
    with col2:
        Glucose = st.text_input('Glucose Level')
    
    with col3:
        BloodPressure = st.text_input('Blood Pressure value')
    
    with col1:
        SkinThickness = st.text_input('Skin Thickness value')
    
    with col2:
        Insulin = st.text_input('Insulin Level')
    
    with col3:
        BMI = st.number_input('BMI value')
    
    with col1:
        DiabetesPedigreeFunction = st.text_input('Diabetes Pedigree Function value')
    
    with col2:
        Age = st.text_input('Age of the Person')
    test =''
      
    if st.button('Diabetes Test Result'):
        diab_prediction = diabetes_model.predict([[Pregnancies, Glucose, BloodPressure, SkinThickness, Insulin, BMI, DiabetesPedigreeFunction, Age]])
        
        if (diab_prediction[0] == 1):
          test = 'The person is diabetic'
        else:
          test = 'The person is not diabetic'
        
    st.success(test)    
 
# Heart Disease Prediction Page
if (selected == 'Heart Disease Prediction'):
    
    # page title
    st.title('Heart Disease Prediction using ML')
    
    col1, col2, col3 = st.columns(3)
    
    with col1:
        age = st.number_input('Age')
        
    with col2:
        sex = st.number_input('Sex')
        
    with col3:
        cp = st.number_input('Chest Pain types')
        
    with col1:
        trestbps = st.number_input('Resting Blood Pressure')
        
    with col2:
        chol = st.number_input('Serum Cholestoral in mg/dl')
        
    with col3:
        fbs = st.number_input('Fasting Blood Sugar > 120 mg/dl')
        
    with col1:
        restecg = st.number_input('Resting Electrocardiographic results')
        
    with col2:
        thalach = st.number_input('Maximum Heart Rate achieved')
        
    with col3:
        exang = st.number_input('Exercise Induced Angina')
        
    with col1:
        oldpeak = st.number_input('ST depression induced by exercise')
        
    with col2:
        slope = st.number_input('Slope of the peak exercise ST segment')
        
    with col3:
        ca = st.number_input('Major vessels colored by flourosopy')
        
    with col1:
        thal = st.number_input('thal: 0 = normal; 1 = fixed defect; 2 = reversable defect')
        
        
     
     
    # code for Prediction
    heart_diagnosis = ''
    
    # creating a button for Prediction
    
    if st.button('Heart Disease Test Result'):
        heart_prediction = heartd_model.predict([[age, sex, cp, trestbps, chol, fbs, restecg,thalach,exang,oldpeak,slope,ca,thal]])                          
        
        if (heart_prediction[0] == 1):
          heart_diagnosis = 'The person is having heart disease'
        else:
          heart_diagnosis = 'The person does not have any heart disease'
        
    st.success(heart_diagnosis)
        
    
    

# Parkinson's Prediction Page
if (selected == "Parkinsons Prediction"):
    
    # page title
    st.title("Parkinson's Disease Prediction using ML")
    
    col1, col2, col3, col4, col5 = st.columns(5)  
    
    with col1:
        fo = st.text_input('MDVP:Fo(Hz)')
        
    with col2:
        fhi = st.text_input('MDVP:Fhi(Hz)')
        
    with col3:
        flo = st.text_input('MDVP:Flo(Hz)')
        
    with col4:
        Jitter_percent = st.text_input('MDVP:Jitter(%)')
        
    with col5:
        Jitter_Abs = st.text_input('MDVP:Jitter(Abs)')
        
    with col1:
        RAP = st.text_input('MDVP:RAP')
        
    with col2:
        PPQ = st.text_input('MDVP:PPQ')
        
    with col3:
        DDP = st.text_input('Jitter:DDP')
        
    with col4:
        Shimmer = st.text_input('MDVP:Shimmer')
        
    with col5:
        Shimmer_dB = st.text_input('MDVP:Shimmer(dB)')
        
    with col1:
        APQ3 = st.text_input('Shimmer:APQ3')
        
    with col2:
        APQ5 = st.text_input('Shimmer:APQ5')
        
    with col3:
        APQ = st.text_input('MDVP:APQ')
        
    
        
    
    
    # code for Prediction
    parkinsons_diagnosis = ''
    
    # creating a button for Prediction    
    if st.button("Parkinson's Test Result"):
        parkinsons_prediction = parki_model.predict([[fo, fhi, flo, Jitter_percent, Jitter_Abs, RAP, PPQ,DDP,Shimmer,Shimmer_dB,APQ3,APQ5,APQ]])                          
        
        if (parkinsons_prediction[0] == 0):
          parkinsons_diagnosis = "The person has Parkinson's disease"
        else:
          parkinsons_diagnosis = "The person does not have Parkinson's disease"
        
    st.success(parkinsons_diagnosis)
    
    
# Breast Cancer Prediction Page
if (selected == "Breast Cancer Prediction"):
    
    # page title
    st.title("Breast Cancer Disease Prediction using ML")
    
    col1, col2, col3 = st.columns(3)  
    
    with col1:
        radius_mean = st.number_input('Mean Radius')
        
    with col2:
        texture_mean = st.number_input('Mean Texture')
        
    with col3:
        perimeter_mean = st.number_input('Mean Perimeter')
        
    with col1:
        area_mean = st.number_input('Mean Area')
        
    with col2:
        smoothness_mean = st.number_input('Mean Smoothness')
        
    with col3:
        compactness_mean = st.number_input('Mean Compactness')
        
    with col1:
        concavity_mean = st.number_input('Mean Concavity')
        
    with col2:
        concavitypoint_mean = st.number_input('Mean Concavity Point')
        
    with col3:
        symmetry_mean = st.number_input('Mean Symmetry')
        
    with col1:
        fractal_dimension_mean = st.number_input('Mean Fractal Dimension')
        
   
        
    
        
    
    
    # code for Prediction
    bc_diagnosis = ''
    
    # creating a button for Prediction    
    if st.button("Breast Cancer Test Result"):
        breastcancer_prediction = breastc_model.predict([[radius_mean,texture_mean,perimeter_mean,area_mean,smoothness_mean,compactness_mean,concavity_mean,concavitypoint_mean,symmetry_mean,fractal_dimension_mean]])                          
        
        if (breastcancer_prediction[0] == 'M'):
          bc_diagnosis = "The person has Malignanat Tumor"
        else:
          bc_diagnosis = "The person has Benign Tumour"
        
    st.success(bc_diagnosis)    